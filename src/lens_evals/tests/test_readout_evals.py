"""Readout evals: pure functions pinned on hand-computed values, the item helpers
of ``readout_eval_items`` (exclusion mask, held-out split, ``load_recipe_items``),
and an end-to-end run of the runner on TinyDecoder with a character-level stub
tokenizer. The pass@k summaries are tested in ``test_readout_eval_scoring.py``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from lens_evals.eval_utils import (
    candidate_token_ids,
    next_token_matches_target,
    surface_forms,
)
from lens_evals.readout_evals.readout_eval_items import (
    DEFAULT_FIT_FRACTION,
    MACRO_EVALS,
    ReadoutEvalItem,
    filter_items_by_correctness,
    item_candidate_token_ids,
    load_readout_eval_items,
    load_recipe_items,
    non_semantic_token_ids,
    split_fitting_items,
)
from lens_evals.readout_evals.readout_evals import (
    ReadoutEvalRunner,
    ReadoutResiduals,
    min_rank_over_candidates,
    readout_position,
)
from workspace_lens.tests.fixtures import (
    make_logit_lens,
    make_tiny_decoder,
    write_readout_eval_json,
)
from workspace_lens.tests.tokenizers import CharTokenizer
from workspace_lens.utils import record_activations

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_READOUT_EVALS_DATA_DIR = REPO_ROOT / "data" / "jlens" / "evaluations"


### PURE-FUNCTION UNIT TESTS


def test_surface_forms() -> None:
    assert surface_forms("brazil") == ["Brazil", " Brazil", "brazil", " brazil"]
    # Capitalised words also get their lowercase surfaces, so casing never
    # penalises a lens in either direction.
    assert surface_forms("Brazil") == ["Brazil", " Brazil", "brazil", " brazil"]


def test_candidate_token_ids_and_synonym_expansion() -> None:
    tokenizer = CharTokenizer()
    # "d" is single-token in every casing (case-insensitive tokenizer); the
    # leading-space surfaces are two tokens and are dropped; ids dedupe.
    assert candidate_token_ids(tokenizer, "d") == [4]
    # Multi-character intermediates have no single-token surface at all.
    assert candidate_token_ids(tokenizer, "xy") == []
    # Order-ops expansion: "5" -> {"5", "five"}; "5" tokenizes to the
    # unknown id 29 (single token), "five" is multi-token under this
    # tokenizer, so the candidate set is just {29}.
    assert candidate_token_ids(tokenizer, "5", expand_order_ops_synonyms=True) == [29]


def test_item_candidate_token_ids() -> None:
    tokenizer = CharTokenizer()
    # order-ops items expand synonyms: "5" -> {"5", "five"} -> only the
    # single-token "5" survives (unknown-char id 29); "d" is not a synonym
    # key and scores as itself (id 4).
    order_ops_item = ReadoutEvalItem("order-ops", "i1", "p", None, ("5", "d"))
    assert item_candidate_token_ids(tokenizer, order_ops_item) == {29, 4}
    # Other evals never expand.
    typo_item = ReadoutEvalItem("typo", "i2", "p", None, ("c",))
    assert item_candidate_token_ids(tokenizer, typo_item) == {3}


def test_next_token_matches_target() -> None:
    assert next_token_matches_target("Atlantic", "Atlantic")
    assert next_token_matches_target(" Atl", "Atlantic")  # leading-space subword
    assert next_token_matches_target("2", "20")  # digit-split subword
    assert not next_token_matches_target("atl", "Atlantic")  # case-sensitive
    assert not next_token_matches_target(" ", "Atlantic")  # whitespace-only
    assert not next_token_matches_target("Pacific", "Atlantic")


def test_readout_position() -> None:
    tokenizer = CharTokenizer()
    input_ids_Int_1S = tokenizer("ab\ncd", return_tensors="pt").input_ids
    # Tokens: [BOS, a, b, \n, c, d] -> final token 5, last newline 3.
    assert readout_position(input_ids_Int_1S, tokenizer, "final_token") == 5
    assert readout_position(input_ids_Int_1S, tokenizer, "last_newline") == 3
    no_newline_Int_1S = tokenizer("abc", return_tensors="pt").input_ids
    with pytest.raises(ValueError, match="no newline"):
        readout_position(no_newline_Int_1S, tokenizer, "last_newline")
    # Rule names other than the eval spec's two are rejected.
    with pytest.raises(ValueError, match="unknown readout rule"):
        readout_position(input_ids_Int_1S, tokenizer, "bogus")


def test_min_rank_over_candidates() -> None:
    logits_V = torch.tensor([5.0, 3.0, 4.0, 1.0, 2.0])
    assert min_rank_over_candidates(logits_V, [0]) == 1
    assert min_rank_over_candidates(logits_V, [3]) == 5
    # Min over candidates: {3, 2} -> best is token 2 (logit 4.0, rank 2).
    assert min_rank_over_candidates(logits_V, [3, 2]) == 2


def test_non_semantic_token_ids_spares_candidates_and_ignores_special_ids() -> None:
    """Under the char tokenizer with vocab 32: BOS (decodes to ""), newline,
    space, "?" (29) and the padding id 31 (decodes to "?") are non-semantic;
    EOS 30 decodes to the word-like "<eos>" and is NOT excluded (special ids
    get no special treatment in the eval mask); an item whose intermediate
    is "?" spares id 29."""
    tokenizer = CharTokenizer()
    assert non_semantic_token_ids(tokenizer, vocab_size=32, items=[]) == [
        0,
        27,
        28,
        29,
        31,
    ]
    punctuation_item = ReadoutEvalItem("typo", "q", "abc", None, ("?",))
    assert non_semantic_token_ids(tokenizer, vocab_size=32, items=[punctuation_item]) == [
        0,
        27,
        28,
        31,
    ]
    # vocab_size bounds the scan: the LM-head width, not len(tokenizer).
    assert non_semantic_token_ids(tokenizer, vocab_size=29, items=[]) == [0, 27, 28]


def test_split_fitting_items_synthetic() -> None:
    items = [ReadoutEvalItem("b", f"b{i}", "p", None, ("x",)) for i in range(5)] + [
        ReadoutEvalItem("a", f"a{i}", "p", None, ("x",)) for i in range(4)
    ]
    fitting, held_out = split_fitting_items(items, seed=0, fit_fraction=0.5)
    # floor(fraction * n) of each eval is in the fitting split; the splits partition the
    # items and keep the input order.
    assert sum(item.eval_slug == "b" for item in fitting) == 2
    assert sum(item.eval_slug == "a" for item in fitting) == 2
    assert {item.name for item in fitting}.isdisjoint({item.name for item in held_out})
    assert len(fitting) + len(held_out) == len(items)
    assert [item.name for item in fitting] == [
        item.name for item in items if item.name in {i.name for i in fitting}
    ]
    # Every eval keeps at least one fitting item, however small the fraction.
    small, _ = split_fitting_items(items, seed=0, fit_fraction=0.05)
    assert sorted(item.eval_slug for item in small) == ["a", "b"]
    # A smaller fraction takes the front of the same permutation.
    assert {item.name for item in small} <= {item.name for item in fitting}
    # The default fraction is the module's.
    default_fitting, _ = split_fitting_items(items, seed=0)
    explicit_fitting, _ = split_fitting_items(items, seed=0, fit_fraction=DEFAULT_FIT_FRACTION)
    assert default_fitting == explicit_fitting
    # Deterministic in the seed, different across seeds (9 items: practically certain).
    again, _ = split_fitting_items(items, seed=0, fit_fraction=0.5)
    assert [item.name for item in again] == [item.name for item in fitting]
    other_seed, _ = split_fitting_items(items, seed=1, fit_fraction=0.5)
    assert [item.name for item in other_seed] != [item.name for item in fitting]
    for bad_fraction in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="fit_fraction"):
            split_fitting_items(items, seed=0, fit_fraction=bad_fraction)
    with pytest.raises(ValueError, match="unique"):
        split_fitting_items(items + items[:1], seed=0)


# A regression pin of split_fitting_items's current output for a sample of (eval, item) keys
# at seed 0 and fit_fraction 0.5 (0 = fitting, 1 = held out), on the 409 items of
# load_recipe_items() minus multihop/half-clock-hours, whose only intermediate "12" has no
# single-token surface under Qwen3.6-27B's digit-splitting tokenizer.
PINNED_SEED0_SPLIT: dict[tuple[str, str], int] = {
    ("multihop", "carnival-ocean"): 1,
    ("multihop", "amazon-language"): 1,
    ("multihop", "super-populous-capital"): 1,
    ("multihop", "dbl-altitude-antonym"): 0,
    ("multilingual", "spanish-opposite-big"): 0,
    ("multilingual", "french-season-summer"): 0,
    ("multilingual", "es-month-after-mar"): 1,
    ("multilingual", "frn-month-after-jun"): 0,
    ("poetry", "couplet-breath-death"): 0,
    ("poetry", "couplet-stone-bone"): 0,
    ("poetry", "couplet-star-far"): 1,
    ("poetry", "couplet-sound-found"): 1,
    ("association", "grief"): 1,
    ("association", "pregnant"): 1,
    ("association", "war-h"): 1,
    ("association", "es-musica"): 0,
    ("typo", "typo-language"): 1,
    ("typo", "typo-government"): 1,
    ("typo", "typo-again"): 0,
    ("typo", "typo-short"): 0,
}
PINNED_ITEMS_PER_EVAL = {
    "multihop": 49,
    "multilingual": 81,
    "poetry": 81,
    "association": 102,
    "typo": 96,
}
PINNED_SKIPPED_ITEM = ("multihop", "half-clock-hours")


@pytest.mark.skipif(
    not REPO_READOUT_EVALS_DATA_DIR.exists(), reason="eval data dir not in this checkout"
)
def test_split_fitting_items_regression_pin() -> None:
    items = [
        item
        for item in load_recipe_items(
            data_dir=str(REPO_READOUT_EVALS_DATA_DIR),
            correctness_csv=str(REPO_READOUT_EVALS_DATA_DIR / "model_correctness.csv"),
        )
        if (item.eval_slug, item.name) != PINNED_SKIPPED_ITEM
    ]
    counts = pd.Series([item.eval_slug for item in items]).value_counts().to_dict()
    assert counts == PINNED_ITEMS_PER_EVAL
    assert [item.eval_slug for item in items] == sorted(
        (item.eval_slug for item in items), key=MACRO_EVALS.index
    )

    fitting_items, held_out_items = split_fitting_items(items, seed=0, fit_fraction=0.5)
    side_of_item = {(item.eval_slug, item.name): 0 for item in fitting_items}
    side_of_item.update({(item.eval_slug, item.name): 1 for item in held_out_items})
    assert len(side_of_item) == len(items)
    for key, pinned_side in PINNED_SEED0_SPLIT.items():
        assert side_of_item[key] == pinned_side, key
    # At fraction 0.5: n // 2 fitting items per eval, the rest held out.
    for eval_slug, num_items in PINNED_ITEMS_PER_EVAL.items():
        assert sum(item.eval_slug == eval_slug for item in fitting_items) == num_items // 2
        assert (
            sum(item.eval_slug == eval_slug for item in held_out_items)
            == num_items - num_items // 2
        )


def test_load_recipe_items(tmp_path: Path) -> None:
    """Five macro evals in MACRO_EVALS order, order-ops left out, correctness
    filter applied (False dropped, NA and True kept)."""
    for slug in (*MACRO_EVALS, "order-ops"):
        write_readout_eval_json(
            tmp_path,
            slug,
            [
                {
                    "name": f"{slug}-1",
                    "prompt": "ab",
                    "intermediates": ["b"],
                    "target": "b",
                },
                {
                    "name": f"{slug}-2",
                    "prompt": "ac",
                    "intermediates": ["c"],
                    "target": "c",
                },
            ],
        )
    correctness_csv = tmp_path / "model_correctness.csv"
    csv_rows = ["eval,item,tiny"]
    for slug in (*MACRO_EVALS, "order-ops"):
        csv_rows.append(f"{slug},{slug}-1,True")
        csv_rows.append(
            f"{slug},{slug}-2," + ("" if slug in ("typo", "association") else "False")
        )
    correctness_csv.write_text("\n".join(csv_rows) + "\n")

    items = load_recipe_items(
        data_dir=str(tmp_path),
        correctness_csv=str(correctness_csv),
        hf_model_name="tiny",
    )
    assert [item.eval_slug for item in items] == [
        "multihop",
        "multilingual",
        "typo",
        "typo",
        "association",
        "association",
        "poetry",
    ]
    assert not any(item.eval_slug == "order-ops" for item in items)
    assert {item.name for item in items if item.eval_slug == "multihop"} == {"multihop-1"}


### END-TO-END ON THE TINY MODEL


def test_runner_end_to_end(tmp_path: Path) -> None:
    model = make_tiny_decoder(CharTokenizer())
    write_readout_eval_json(
        tmp_path,
        "poetry",
        [{"name": "p1", "prompt": "ab\ncd", "intermediates": ["d", "xy"]}],
    )
    write_readout_eval_json(
        tmp_path,
        "typo",
        [{"name": "t1", "prompt": "abc", "intermediates": ["c"]}],
    )
    items = load_readout_eval_items(str(tmp_path), slugs=["poetry", "typo"])
    assert [item.name for item in items] == ["p1", "t1"]

    lens = make_logit_lens([0, 2], tmp_path, checkpoint_name="readout-evals")
    runner = ReadoutEvalRunner(model, {"logit": lens}, max_seq_len=32)
    assert runner.layers == [0, 2]

    correctness_df = runner.grade_model_correctness(items, hf_model_name="tiny")
    assert list(correctness_df.columns) == ["eval", "item", "tiny"]
    # poetry grades against its rhyme-word intermediate; typo is ungradeable.
    assert correctness_df.set_index("eval").loc["poetry", "tiny"] in (True, False)
    assert pd.isna(correctness_df.set_index("eval").loc["typo", "tiny"])

    ranks_df = runner.run(items)
    assert list(ranks_df.columns) == [
        "eval",
        "lens",
        "item",
        "intermediate",
        "layer",
        "rank",
        "n_candidates",
    ]
    # "xy" has no single-token surface: dropped and recorded.
    assert runner.dropped_intermediates == [("poetry", "p1", "xy")]
    # 2 scoreable intermediates x 1 lens x 2 layers.
    assert len(ranks_df) == 4
    assert set(ranks_df["layer"]) == {0, 2}
    assert ranks_df["rank"].between(1, 32).all()


def test_runner_ranks_match_hand_readout(tmp_path: Path) -> None:
    """run() equals a hand readout of the lens at the spec's position
    (poetry: last newline; typo: final token), rank = 1 + strictly-greater."""
    model = make_tiny_decoder(CharTokenizer())
    # poetry "ab \ncd.": 0 BOS, 1 a, 2 b, 3 ' ', 4 '\n', 5 c, 6 d, 7 '.' -> 4.
    # typo "abc. ": 0 BOS, 1 a, 2 b, 3 c, 4 '.', 5 ' ' -> 5.
    write_readout_eval_json(
        tmp_path,
        "poetry",
        [{"name": "p1", "prompt": "ab \ncd.", "intermediates": ["d"]}],
    )
    write_readout_eval_json(
        tmp_path, "typo", [{"name": "t1", "prompt": "abc. ", "intermediates": ["c"]}]
    )
    items = load_readout_eval_items(str(tmp_path), slugs=["poetry", "typo"])
    lens = make_logit_lens([0, 2], tmp_path, checkpoint_name="readout-readout")
    ranks_df = ReadoutEvalRunner(model, {"logit": lens}, max_seq_len=32).run(items)
    expected_positions = {"poetry": 4, "typo": 5}
    for item in items:
        _, activations_L_dict_SN = record_activations(model, item.prompt, 32, [0, 2])
        position = expected_positions[item.eval_slug]
        candidate_ids = candidate_token_ids(model.tokenizer, item.intermediates[0])
        for layer in (0, 2):
            residual_1N = activations_L_dict_SN[layer][[position]].float()
            logits_V = model.unembed(lens.transport(residual_1N, layer))[0]
            expected_rank = min_rank_over_candidates(logits_V, candidate_ids)
            actual_rank = ranks_df[
                (ranks_df["eval"] == item.eval_slug) & (ranks_df["layer"] == layer)
            ]["rank"].item()
            assert actual_rank == expected_rank


def test_excluded_token_ids_rerank(tmp_path: Path) -> None:
    """Excluded tokens are forced to -inf before ranking: the rank equals a
    hand rank over the masked logits, and excluding every non-candidate makes
    the candidate rank 1 — an excluded token never outranks anything."""
    model = make_tiny_decoder(CharTokenizer())
    write_readout_eval_json(
        tmp_path, "typo", [{"name": "t1", "prompt": "abc", "intermediates": ["z"]}]
    )
    items = load_readout_eval_items(str(tmp_path), slugs=["typo"])
    lens = make_logit_lens([0, 2], tmp_path, checkpoint_name="readout-exclude")
    z_id = 26
    plain_df = ReadoutEvalRunner(model, {"logit": lens}, max_seq_len=32).run(items)

    # Exclude the plain top-1 token at every layer (never the candidate).
    _, activations_L_dict_SN = record_activations(model, items[0].prompt, 32, [0, 2])
    top1_ids: set[int] = set()
    for layer in (0, 2):
        residual_1N = activations_L_dict_SN[layer][[-1]].float()
        top1_ids.add(int(model.unembed(lens.transport(residual_1N, layer))[0].argmax()))
    top1_ids -= {z_id}
    assert top1_ids
    excluded_runner = ReadoutEvalRunner(
        model, {"logit": lens}, max_seq_len=32, excluded_token_ids=sorted(top1_ids)
    )
    excluded_df = excluded_runner.run(items)
    for layer in (0, 2):
        residual_1N = activations_L_dict_SN[layer][[-1]].float()
        logits_V = model.unembed(lens.transport(residual_1N, layer))[0].float()
        logits_V[sorted(top1_ids)] = float("-inf")
        expected_rank = min_rank_over_candidates(logits_V, [z_id])
        assert excluded_df[excluded_df["layer"] == layer]["rank"].item() == expected_rank
    # Removing tokens can only improve the candidate's rank.
    assert (excluded_df["rank"] <= plain_df["rank"]).all()

    all_but_z = [token_id for token_id in range(32) if token_id != z_id]
    only_z_scoreable = ReadoutEvalRunner(
        model, {"logit": lens}, max_seq_len=32, excluded_token_ids=all_but_z
    )
    assert (only_z_scoreable.run(items)["rank"] == 1).all()


def test_excluded_token_ids_all_candidates_excluded_raises(tmp_path: Path) -> None:
    model = make_tiny_decoder(CharTokenizer())
    write_readout_eval_json(
        tmp_path, "typo", [{"name": "t1", "prompt": "abc", "intermediates": ["z"]}]
    )
    items = load_readout_eval_items(str(tmp_path), slugs=["typo"])
    runner = ReadoutEvalRunner(
        model,
        {"logit": make_logit_lens([0, 2], tmp_path, checkpoint_name="readout-exclude-all")},
        max_seq_len=32,
        excluded_token_ids=[26],
    )
    with pytest.raises(ValueError, match="excluded_token_ids"):
        runner.run(items)


def test_filter_items_by_correctness(tmp_path: Path) -> None:
    """Keeps True and NA items, drops False, warns-and-drops ungraded items,
    and rejects a CSV without the model's column."""
    write_readout_eval_json(
        tmp_path,
        "typo",
        [
            {"name": "kept-true", "prompt": "abc", "intermediates": ["c"]},
            {"name": "dropped-false", "prompt": "abd", "intermediates": ["d"]},
            {"name": "kept-na", "prompt": "abe", "intermediates": ["e"]},
            {"name": "ungraded", "prompt": "abf", "intermediates": ["f"]},
        ],
    )
    items = load_readout_eval_items(str(tmp_path), slugs=["typo"])
    correctness_csv = tmp_path / "model_correctness.csv"
    correctness_csv.write_text(
        "eval,item,tiny\ntypo,kept-true,True\ntypo,dropped-false,False\ntypo,kept-na,\n"
    )
    filtered = filter_items_by_correctness(
        items, hf_model_name="tiny", correctness_csv=str(correctness_csv)
    )
    assert [item.name for item in filtered] == ["kept-true", "kept-na"]
    with pytest.raises(ValueError, match="no column"):
        filter_items_by_correctness(
            items, hf_model_name="other-model", correctness_csv=str(correctness_csv)
        )


### READOUT RESIDUALS


def _residual_items() -> list[ReadoutEvalItem]:
    return [
        # poetry: read out at the newline; "xy" is dropped (multi-token).
        ReadoutEvalItem("poetry", "p1", "ab\ncd", None, ("d", "xy")),
        # typo: read out at the final token; two scoreable intermediates.
        ReadoutEvalItem("typo", "t1", "abc", None, ("c", "a")),
        # No scoreable intermediate at all: skipped entirely.
        ReadoutEvalItem("typo", "t2", "abc", None, ("xy",)),
    ]


def test_record_readout_residuals_agrees_with_runner(tmp_path: Path) -> None:
    model = make_tiny_decoder(CharTokenizer())
    items = _residual_items()
    lens = make_logit_lens([0, 2], tmp_path, checkpoint_name="readout-residuals")
    residuals = ReadoutEvalRunner(model, {}, layers=[2, 0], max_seq_len=32).record_residuals(
        items, hf_model_name="tiny"
    )
    # A lens-less runner records, but has no lens to take its layers from.
    with pytest.raises(ValueError, match="layers is required"):
        ReadoutEvalRunner(model, {}, max_seq_len=32)
    assert residuals.layers == [0, 2]
    assert residuals.item_keys == [("poetry", "p1"), ("typo", "t1")]
    assert residuals.candidate_ids_P_list == [{"d": [4]}, {"c": [3], "a": [1]}]
    assert residuals.hf_model_name == "tiny" and residuals.max_seq_len == 32
    for layer in (0, 2):
        residuals_PN = residuals.residuals_L_dict_PN[layer]
        assert residuals_PN.shape == (2, 8)
        assert residuals_PN.dtype == torch.float32 and residuals_PN.device.type == "cpu"

    # The recorded vectors are the residuals at the runner's positions
    # (poetry "ab\ncd": [BOS a b \n c d] -> 3; typo "abc": final token 3):
    # ranking them through the lens reproduces run() for every pair and layer.
    runner = ReadoutEvalRunner(model, {"logit": lens}, max_seq_len=32)
    ranks_df = runner.run(items)
    assert runner.dropped_intermediates == [
        ("poetry", "p1", "xy"),
        ("typo", "t2", "xy"),
    ]
    ranks_by_key = ranks_df.set_index(["item", "intermediate", "layer"])["rank"]
    assert len(ranks_by_key) == 3 * 2
    for item_idx, (_, item_name) in enumerate(residuals.item_keys):
        for layer in residuals.layers:
            residual_1N = residuals.residuals_L_dict_PN[layer][[item_idx]]
            logits_V = model.unembed(lens.transport(residual_1N, layer))[0].float()
            for intermediate, candidate_ids in residuals.candidate_ids_P_list[
                item_idx
            ].items():
                assert ranks_by_key[
                    (item_name, intermediate, layer)
                ] == min_rank_over_candidates(logits_V, candidate_ids)
    # And they are literally the residual at that position.
    _, activations_L_dict_SN = record_activations(model, "ab\ncd", 32, [0, 2])
    assert torch.equal(
        residuals.residuals_L_dict_PN[2][0], activations_L_dict_SN[2][3].float()
    )


def test_readout_residuals_cache_round_trip_and_key_mismatch(tmp_path: Path) -> None:
    model = make_tiny_decoder(CharTokenizer())
    items = _residual_items()
    cache_path = tmp_path / "readout_residuals.pt"
    runner = ReadoutEvalRunner(model, {}, layers=[0, 2], max_seq_len=32)
    recorded = runner.record_residuals(items, cache_path=cache_path, hf_model_name="tiny")
    assert cache_path.exists()

    loaded = ReadoutResiduals.load(cache_path)
    assert loaded.cache_header() == recorded.cache_header()
    assert loaded.candidate_ids_P_list == recorded.candidate_ids_P_list
    for layer in (0, 2):
        assert torch.equal(
            loaded.residuals_L_dict_PN[layer], recorded.residuals_L_dict_PN[layer]
        )

    # A matching request loads the cache instead of re-recording: perturb the
    # file and see the perturbed values come back.
    loaded.residuals_L_dict_PN[0][0, 0] = 123.0
    loaded.save(cache_path)
    from_cache = runner.record_residuals(items, cache_path=cache_path, hf_model_name="tiny")
    assert from_cache.residuals_L_dict_PN[0][0, 0].item() == 123.0

    # Any header difference raises rather than silently reusing the cache.
    mismatches = [
        {"layers": [0], "max_seq_len": 32, "hf_model_name": "tiny", "items": items},
        {"layers": [0, 2], "max_seq_len": 16, "hf_model_name": "tiny", "items": items},
        {"layers": [0, 2], "max_seq_len": 32, "hf_model_name": "other", "items": items},
        {
            "layers": [0, 2],
            "max_seq_len": 32,
            "hf_model_name": "tiny",
            "items": items[:1],
        },
    ]
    for mismatch in mismatches:
        mismatched_runner = ReadoutEvalRunner(
            model,
            {},
            layers=mismatch["layers"],
            max_seq_len=mismatch["max_seq_len"],
        )
        with pytest.raises(ValueError, match="recorded for another"):
            mismatched_runner.record_residuals(
                mismatch["items"],
                cache_path=cache_path,
                hf_model_name=mismatch["hf_model_name"],
            )
    # Test models carry no HF name: it must be given.
    with pytest.raises(ValueError, match="no HF name"):
        runner.record_residuals(items)
