# Implementation Plan

[Overview]
PR 리뷰 5건을 벤치마크 타깃 필터링, 디바이스 트래커 수명주기, batched TPS 토큰 산정, batched phase metric 처리, Qwen3-VL generation kwargs 전달까지 각각 독립적으로 수정하고 atomic commit 후 현재 브랜치를 푸시한다.

이번 변경의 범위는 `benchmark/transformers`의 다중 모델 벤치마크 스크립트, `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py`의 TPS 측정 핵심 로직, `mblt_model_zoo/cli/tps.py`의 단일 TPS CLI phase metric 처리, Qwen3-VL 모델 generation hook, 그리고 해당 동작을 검증하는 `tests/transformers` 단위 테스트로 제한한다.

핵심 접근은 기존 public CLI 옵션과 결과 JSON/CSV schema를 유지하면서 잘못된 skip, tracker leak, decode token 과대계상, 관측 불가능한 batch phase metric 오염, `prefill_chunk_size` 전달 누락을 최소 침습적으로 보정하는 것이다. 각 리뷰 항목은 독립적인 논리 변경이므로 코드와 테스트를 같은 atomic commit에 묶고, 마지막에 targeted validation과 pre-commit을 통과시킨 뒤 push한다.

작업 시 repository-wide `AGENTS.md` 규칙을 따른다. Python 명령은 프로젝트에 `.venv`, `uv.lock`, `pyproject.toml`이 있으므로 `uv run ...`을 우선 사용한다. `mblt_model_zoo/hf_transformers` 경로는 Ruff exclude 영역이므로 기존 로컬 스타일을 유지하되, 새로 수정하는 함수 signature에는 타입을 명확히 하고 새 helper에는 Google-style docstring을 추가한다.

[Types]
기존 dataclass와 public result schema는 유지하되 batch filtering과 phase metric 처리에서 사용하는 내부 의미를 명확히 한다.

- `benchmark.transformers.benchmark_text_generation_models.TextBenchmarkTarget`.
  - 기존 필드 유지: `model_id: str`, `revision_candidates: list[str | None]`, `label: str`, `base: str`, `mxq_path: str | None`, `max_batch_size: int`.
  - `max_batch_size`는 필터 통과 후 항상 `int >= 1`로 유지한다.
  - config metadata가 없고 `batch_mode == "non_batch"`인 target은 `max_batch_size=1`로 정규화한다.
  - config metadata가 없고 `batch_mode == "batch"`인 target은 explicit batch capability가 없으므로 제외한다.

- Batch mode constants.
  - `_BATCH_MODE_BATCH = "batch"`는 explicit `max_batch_size > 1`만 허용한다.
  - `_BATCH_MODE_NON_BATCH = "non_batch"`는 explicit `max_batch_size == 1` 또는 missing metadata fallback `1`을 허용한다.

- `mblt_model_zoo.hf_transformers.utils.benchmark_utils.SingleMeasurement`.
  - 구조 변경 없음.
  - `num_decode`는 실제 decode throughput 계산에 사용된 generated decode token 수를 의미한다.
  - VLM batched LLM path에서는 `outputs.shape[1]` 전체 길이가 아니라 `max(outputs.shape[1] - seq_len, 0)`로 계산한 generated token 수에서 first-token/prefill token 1개를 제외한 값으로 저장한다.
  - Non-fake batched text generation에서는 phase tracker callback이 호출되지 않으므로 phase별 device metric 필드는 caller에서 채우지 않는다.

- Qwen3-VL generation kwargs.
  - `MobilintQwen3VLForConditionalGeneration.prepare_inputs_for_generation()`에 keyword-only `prefill_chunk_size: Optional[int] = None`를 추가한다.
  - `prefill_chunk_size`가 `None`이 아니면 returned `model_inputs["prefill_chunk_size"]`에 보존한다.
  - `count_npu_time: bool` 전달 동작은 유지한다.

- Device tracker protocol/result schema.
  - `DeviceTracker` protocol과 JSON/CSV keys는 변경하지 않는다.
  - 관측 가능한 phase boundary가 없는 fixed-size batched text measurement에서는 phase device metric extraction을 생략하여 기존 key에는 `None` 또는 빈 summary가 남도록 한다.
  - VLM fixed measure의 single whole-run tracker는 vision과 LLM 측정 전체를 감싸되, 예외 발생 시 반드시 stop된다.

[Files]
수정은 벤치마크 스크립트, TPS utility, Qwen3-VL hook, 테스트, 계획 문서에 한정한다.

- New files to be created.
  - `implementation_plan.md`: 이 구현 계획 문서.

- Existing files to be modified.
  - `benchmark/transformers/benchmark_text_generation_models.py`.
    - Missing `max_batch_size` metadata를 `--non-batch`에서는 `1`로 처리하고 `--batch`에서는 skip하도록 target filtering을 수정한다.
    - Fixed text `measure` path에서 `batch_size > 1`이면 phase tracker callbacks/extraction을 비활성화한다.
    - 기존 `sweep` path의 outer prefill/decode phase callbacks는 phase boundary가 있으므로 유지한다.
  - `benchmark/transformers/benchmark_image_text_to_text_models.py`.
    - `_run_measure()`에서 tracker가 `measure_vision()` 전에 start된 뒤 `measure_vision()` 또는 `measure_llm_full()` 어디서 예외가 나도 `_stop_tracker_safe(tracker)`가 실행되도록 try/finally 범위를 확장한다.
    - Shared text target filter 변경을 VLM target collection에도 그대로 적용한다.
  - `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py`.
    - `TPSMeasurer._measure_batch_generate()`에서 non-fake batched generation의 unobservable phase callbacks를 호출하지 않도록 수정한다.
    - `VLMTPSMeasurer._measure_llm_once()` batched path의 generated/decode token count를 prompt prefix 길이를 제외하도록 수정한다.
  - `mblt_model_zoo/cli/tps.py`.
    - Fixed text `_run_text_measure()`에서 `batch_size > 1`이면 phase callbacks를 전달하지 않고 phase metric extraction/enrichment를 건너뛰어 batch mode phase metrics 오염을 방지한다.
    - Text sweep path는 `measure_full()`의 outer phase boundary가 존재하므로 변경하지 않는다.
  - `mblt_model_zoo/hf_transformers/models/qwen3_vl/modeling_qwen3_vl.py`.
    - Qwen3-VL `prepare_inputs_for_generation()` decorator/signature에 `prefill_chunk_size`를 추가하고 returned `model_inputs`에 보존한다.
  - `tests/transformers/test_benchmark_transformers_cli.py`.
    - Missing `max_batch_size` fallback과 `--batch` explicit metadata requirement를 검증한다.
    - VLM fixed measure에서 vision 측정 실패 시 tracker stop이 보장되는지 검증한다.
  - `tests/transformers/test_cli_tps.py`.
    - Batched VLM LLM decode token count가 prompt length를 제외하는지 검증한다.
    - Non-fake batched text generate에서 unobservable phase callbacks가 호출되지 않는지 검증한다.
    - Qwen3-VL `prepare_inputs_for_generation` signature와 model input preservation이 `prefill_chunk_size`를 포함하는지 검증한다.

- Files to be deleted or moved.
  - 없음.

- Configuration file updates.
  - 없음. `pyproject.toml`, `uv.lock`, dependency manifests는 변경하지 않는다.

[Functions]
새 helper를 최소로 추가하고 기존 measurement/filtering/generation hook 함수를 수정한다.

- New functions.
  - `benchmark/transformers/benchmark_text_generation_models.py::_resolve_filter_max_batch_size(max_batch_size: int | None, batch_mode: str) -> int | None`.
    - Purpose: config에서 읽은 raw `max_batch_size` 결과를 batch-mode filtering에 사용할 effective value로 변환한다.
    - Rules: explicit value는 그대로 사용, missing value는 `non_batch`에서 `1`, `batch`에서 `None`, unsupported mode는 `ValueError`.
    - This helper makes missing metadata policy testable and keeps `_filter_text_targets_by_batch_mode()` readable.

- Modified functions.
  - `benchmark/transformers/benchmark_text_generation_models.py::_filter_text_targets_by_batch_mode(...) -> list[TextBenchmarkTarget]`.
    - `_resolve_config_max_batch_size()` 결과를 `_resolve_filter_max_batch_size()`로 정규화한다.
    - `batch_mode == "batch"`이고 metadata가 missing이면 기존처럼 skip하되 message를 batch explicit metadata requirement로 유지한다.
    - `batch_mode == "non_batch"`이고 metadata가 missing이면 skip하지 않고 `TextBenchmarkTarget.max_batch_size=1`로 append한다.
  - `benchmark/transformers/benchmark_text_generation_models.py::_run_measure(args: argparse.Namespace) -> int`.
    - `batch_size > 1`인 fixed text measurement에서는 `on_prefill_start`, `on_prefill_end`, `on_decode_start`, `on_decode_end`를 `None`으로 전달한다.
    - `tracker_prefill`/`tracker_decode` metric extraction과 `_enrich_single_run_device()` 호출도 `batch_size == 1`일 때만 수행한다.
  - `benchmark/transformers/benchmark_image_text_to_text_models.py::_run_measure(args: argparse.Namespace) -> int`.
    - Repeat loop에서 `tracker.start()` 이후 `measure_vision()`과 `measure_llm_full()`을 동일한 try/finally 안에 넣는다.
    - `vision_latency`, `vision_fps`, `llm_result` 사용은 try block 성공 이후에만 이어지므로 기존 outer except가 실패 target을 처리한다.
  - `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py::TPSMeasurer._measure_batch_generate(...) -> SingleMeasurement`.
    - `fake_prefill=False` and `batch_size > 1`에서는 phase callbacks를 호출하지 않는다.
    - `fake_prefill=True`에서는 기존 decode-only semantics를 유지한다.
    - 후행 `on_decode_start()` after `generate()` 호출을 제거하여 decode tracker가 completed generation 뒤에 시작되는 문제를 없앤다.
  - `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py::VLMTPSMeasurer._measure_llm_once(...) -> SingleMeasurement`.
    - Batched path에서 tensor output 길이를 `output_len`으로 읽고 `generated_per_row = max(output_len - seq_len, 0)`로 계산한다.
    - `decode_count = max(generated_per_row - 1, 0)`로 계산하며 tensor가 아닌 output fallback은 기존 `num_decode + 1` assumptions를 유지한다.
    - `total_decode_tokens`, `decode_tps`, `avg_total_decode_token_latency`, `avg_npu_decode_token_latency`, returned `num_decode`는 corrected `decode_count`를 사용한다.
  - `mblt_model_zoo/cli/tps.py::_run_text_measure(args: argparse.Namespace) -> int`.
    - `phase_callbacks_supported = batch_size == 1` 같은 local boolean을 사용해 fixed batch text measurement의 phase tracker callbacks/extraction/enrichment를 비활성화한다.
  - `mblt_model_zoo/hf_transformers/models/qwen3_vl/modeling_qwen3_vl.py::MobilintQwen3VLForConditionalGeneration.prepare_inputs_for_generation(...)`.
    - Decorator를 `@with_mobilint_generation_signature(Qwen3VLForConditionalGeneration.prepare_inputs_for_generation, "count_npu_time", "prefill_chunk_size")`로 변경한다.
    - Signature에 `prefill_chunk_size: Optional[int] = None`를 추가한다.
    - `prefill_chunk_size is not None`이면 `model_inputs["prefill_chunk_size"] = prefill_chunk_size`를 수행한다.

- Removed functions.
  - 없음.

[Classes]
새 클래스는 추가하지 않고 기존 measurement/model classes의 일부 method만 수정한다.

- New classes.
  - 없음.

- Modified classes.
  - `mblt_model_zoo.hf_transformers.utils.benchmark_utils.TPSMeasurer`.
    - `_measure_batch_generate()` callback semantics를 수정한다.
    - Public constructor와 public `measure()`/`measure_full()` signature는 변경하지 않는다.
  - `mblt_model_zoo.hf_transformers.utils.benchmark_utils.VLMTPSMeasurer`.
    - `_measure_llm_once()` batched decode count 산식을 수정한다.
    - Public constructor와 public `measure_*()` signatures는 변경하지 않는다.
  - `mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl.MobilintQwen3VLForConditionalGeneration`.
    - `prepare_inputs_for_generation()` hook만 수정한다.
    - `forward()`와 `_get_cache()` behavior는 변경하지 않는다.
  - Test-only doubles in `tests/transformers/test_cli_tps.py`.
    - 필요한 경우 batched VLM/text generation 테스트를 위해 lightweight dummy model/tracker class를 추가한다.

- Removed classes.
  - 없음.

[Dependencies]
새 dependency나 version 변경은 없다.

기존 프로젝트는 `pyproject.toml`과 `uv.lock`이 있으며 `.venv`도 존재하므로 validation은 `uv run` 기반으로 수행한다. PR 리뷰 수정은 Python standard library, PyTorch tensor shape handling, 기존 pytest/monkeypatch utilities만 사용한다.

`mblt-tracker`, `transformers`, `torch`, `pytest`, `ruff`, `pre-commit` dependency는 기존 선언을 그대로 사용한다. Qwen3-VL tests는 현재 테스트 파일들이 이미 사용하는 compatibility guard 또는 import 조건을 존중한다.

[Testing]
Targeted pytest와 lint/pre-commit으로 리뷰 항목별 regression을 검증한다.

- Unit tests to add or modify in `tests/transformers/test_benchmark_transformers_cli.py`.
  - `test_text_target_filtering_treats_missing_max_batch_size_as_non_batch`.
    - `_resolve_config_max_batch_size()`가 `None`을 반환하는 upstream/original-like target을 구성한다.
    - `batch_mode="non_batch"`에서는 target이 포함되고 `max_batch_size == 1`인지 확인한다.
    - `batch_mode="batch"`에서는 같은 target이 제외되는지 확인한다.
  - Existing `test_text_target_filtering_by_batch_mode`는 explicit `1`, explicit `4`, GGUF skip behavior가 유지되도록 필요 시 보강한다.
  - `test_vlm_measure_stops_tracker_when_vision_measure_fails`.
    - `_run_measure()`에 fake pipeline, fake `VLMTPSMeasurer`, fake tracker를 주입한다.
    - fake `measure_vision()`이 measured repeat에서 예외를 던지게 한다.
    - `_run_measure()`가 outer except로 target을 skip하더라도 tracker `stop()`이 호출되었음을 확인한다.

- Unit tests to add or modify in `tests/transformers/test_cli_tps.py`.
  - `test_vlm_batched_llm_decode_count_subtracts_prompt_length`.
    - `VLMTPSMeasurer._measure_llm_once()` batched branch를 fake VLM model로 호출한다.
    - `inputs_embeds.shape == (batch_size, seq_len, hidden)`이고 fake `generate()`가 shape `(batch_size, seq_len + num_decode + 1)` tensor를 반환하게 한다.
    - returned `SingleMeasurement.num_decode == num_decode`이고 `decode_tps` 계산 denominator/numerator가 prompt length를 포함하지 않는지 확인한다.
  - `test_text_batched_generate_disables_unobservable_phase_callbacks`.
    - `TPSMeasurer._measure_batch_generate()` 또는 `TPSMeasurer.measure(batch_size=2)`를 fake generate model로 호출한다.
    - `fake_prefill=False` path에서 `on_decode_start`가 generate 완료 후 호출되지 않고, phase callbacks list가 비어 있거나 documented disabled pattern과 일치하는지 확인한다.
  - `test_qwen3_vl_prepare_inputs_preserves_prefill_chunk_size`.
    - Qwen3-VL hook signature에 `prefill_chunk_size`가 존재하는지 확인한다.
    - Monkeypatch 또는 lightweight dummy를 사용해 `prepare_inputs_for_generation(..., count_npu_time=True, prefill_chunk_size=64)` 반환 dict에 두 key가 보존되는지 확인한다.
  - Existing `test_mobilint_generation_hooks_accept_count_npu_time`는 Qwen3-VL prepare hook의 new parameter와 충돌하지 않는지 유지한다.

- Validation commands.
  - `uv run pytest tests/transformers/test_benchmark_transformers_cli.py tests/transformers/test_cli_tps.py`.
  - `uv run ruff check benchmark/transformers/benchmark_text_generation_models.py benchmark/transformers/benchmark_image_text_to_text_models.py mblt_model_zoo/cli/tps.py tests/transformers/test_benchmark_transformers_cli.py tests/transformers/test_cli_tps.py`.
  - `uv run pre-commit run --files benchmark/transformers/benchmark_text_generation_models.py benchmark/transformers/benchmark_image_text_to_text_models.py mblt_model_zoo/hf_transformers/utils/benchmark_utils.py mblt_model_zoo/cli/tps.py mblt_model_zoo/hf_transformers/models/qwen3_vl/modeling_qwen3_vl.py tests/transformers/test_benchmark_transformers_cli.py tests/transformers/test_cli_tps.py implementation_plan.md`.
  - If Qwen3-VL import depends on a Transformers version not present in the active `.venv`, run the same targeted tests that are compatible and document the environment limitation before committing.

[Implementation Order]
리뷰 항목 순서대로 수정, 테스트, atomic commit을 반복한 뒤 최종 validation과 push를 수행한다.

1. Confirm clean working tree and current branch.
   - Run `git status --short | cat` and verify no unrelated local edits are present.
   - Continue on current branch `beomsu/update-benchmark` unless the user instructs otherwise.

2. Fix review item 1: missing `max_batch_size` should not drop original/non-batch targets.
   - Modify `benchmark/transformers/benchmark_text_generation_models.py` with `_resolve_filter_max_batch_size()` and update `_filter_text_targets_by_batch_mode()`.
   - Add/update tests in `tests/transformers/test_benchmark_transformers_cli.py`.
   - Run targeted test for this behavior.
   - Commit with `fix: keep untagged targets non-batch`.

3. Fix review item 2: stop VLM tracker when vision measurement fails.
   - Modify `benchmark/transformers/benchmark_image_text_to_text_models.py::_run_measure()` try/finally scope.
   - Add regression test in `tests/transformers/test_benchmark_transformers_cli.py`.
   - Run targeted test for tracker stop behavior.
   - Commit with `fix: stop VLM tracker on failure`.

4. Fix review item 3: subtract prompt length from batched VLM decode token count.
   - Modify `mblt_model_zoo/hf_transformers/utils/benchmark_utils.py::VLMTPSMeasurer._measure_llm_once()` batched branch.
   - Add regression test in `tests/transformers/test_cli_tps.py`.
   - Run targeted test for batched VLM decode count.
   - Commit with `fix: count VLM batch decode tokens`.

5. Fix review item 4: prevent post-generation decode tracker start in batched text measurement.
   - Modify `TPSMeasurer._measure_batch_generate()` callback handling.
   - Modify fixed text measurement call sites in `mblt_model_zoo/cli/tps.py::_run_text_measure()` and `benchmark/transformers/benchmark_text_generation_models.py::_run_measure()` to disable phase metric extraction for `batch_size > 1`.
   - Add regression test in `tests/transformers/test_cli_tps.py`.
   - Run targeted test for callback behavior.
   - Commit with `fix: disable batch phase trackers`.

6. Fix review item 5: preserve Qwen3-VL `prefill_chunk_size` through generation preparation.
   - Modify `mblt_model_zoo/hf_transformers/models/qwen3_vl/modeling_qwen3_vl.py::MobilintQwen3VLForConditionalGeneration.prepare_inputs_for_generation()`.
   - Add/update tests in `tests/transformers/test_cli_tps.py`.
   - Run targeted Qwen3-VL hook test.
   - Commit with `fix: keep Qwen3-VL chunk size`.

7. Run final targeted validation.
   - Run `uv run pytest tests/transformers/test_benchmark_transformers_cli.py tests/transformers/test_cli_tps.py`.
   - Run targeted `uv run ruff check ...` for Ruff-covered touched files.
   - Run `uv run pre-commit run --files ...` for all touched files.
   - Fix any failures with additional focused commits or amend the relevant atomic commit only if no push has happened yet and history remains clean.

8. Review final diff and commit history.
   - Run `git status --short | cat`, `git log --oneline -5 | cat`, and targeted `git diff --stat HEAD~5..HEAD | cat`.
   - Ensure there are no uncommitted changes and each commit maps to one review item.

9. Push the branch.
   - Run `git push origin beomsu/update-benchmark` after all commits and validations pass.
   - Report pushed branch, commit list, and validation results to the user.