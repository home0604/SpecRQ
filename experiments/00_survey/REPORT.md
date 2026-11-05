# Phase 1: 코드 조사 및 모델 다운로드 (2026-11-05)

실행: executor agent / 검증: supervisor. [V] 직접 확인, [I] 추론·미검증.
Supervisor 재확인 완료 항목: Qwen revision·파일, DbD α 하드코딩(`aq_engine.py:149`), prefix-k forward(`src/aq.py:200`),
DbD 사본에 Hadamard 없음, DbD requirements(transformers==4.40.1, datasets==2.19.0), AQLM `CodeKx8` 인스턴스 K=1,2뿐.

## 코드베이스

| | Drop-by-Drop (2606.12876) | AQLM | VQ-LLM (2503.02236) | INT4 Marlin |
|---|---|---|---|---|
| 코드 | [V] anonymous.4open.science/r/DropbyDrop2026 → `/data/lsh/src/DropbyDrop2026` (git 이력 없음, 보존 필요) | [V] `/data/lsh/src/AQLM` @ e79a896 | [V] 공개 코드 미발견 (`SpecRQ/VQLLM`과 무관) | [V] `/data/lsh/src/marlin` @ 1f25790 |
| License | Apache-2.0 (AQLM과 동일 파일) | Apache-2.0 | — | Apache-2.0 |
| 베이스 | AQLM; 변경 파일 `aq_engine.py`, `src/aq.py`, `src/utils.py`, `quantize.sh` | — | — | — |
| Nested 학습 | [V] Σα_k·tr(ΔW_k XXᵀ ΔW_kᵀ), α=[0,0,.5,0,.5] 하드코딩. Beam search는 L^(M)만, block fine-tune은 k=M 출력만 | 없음 (AQ + PV-tuning) | — | — |
| Hadamard | 논문 Alg.1에는 있으나 **코드에 없음** | — | — | — |
| 모델 | 논문: Qwen2.5-0.5B~32B-Instruct, Mistral-7B-v0.3, Meta-Llama-3-8B (Llama-3.1 없음) | llama/qwen2 등 | AQLM-3(d8,4096,2) 등, RTX 4090 | FP16×INT4만 (bf16 없음) |
| 커널 | codeNx8 LUT (App. I.3), **미공개** | Code1x16, Code2x8, CodeKx8 (K=1,2만 인스턴스), Triton fallback; interleaved code layout | 설계 참고용 | 빌드 미시도 |
| env 호환 | transformers 4.40.1 기준 → 5.3.0에서 layer 호출(position_embeddings) 깨질 가능성 [I] | nvcc: `/usr/local/cuda-12.1/bin` (PATH 밖) | — | vLLM/GPTQModel 최신은 torch 2.14 요구 |

DbD 논문 Qwen2.5-0.5B-Instruct 참고값: WikiText2 PPL, Had.+35W 5bit 13.36 / 3bit 22.05.

## 추천 출발점 [I]
- 학습: DbD 사본을 vendoring하고 다음을 추가 — α CLI 노출, Hadamard 구현, transformers 호환, (연구) beam search·fine-tune에 k=3 항 추가.
- 커널: AQLM CodeKx8에서 출발해 K=3/5 → DbD식 LUT → planar(codebook-major) code layout으로 draft가 k/L만 읽게. INT4 baseline은 standalone Marlin.
- env 후보: 새 env `nestvq` (py3.11, torch 2.4.1 cu121, transformers ~4.46 — 검증 필요, ninja, CUDA_HOME=/usr/local/cuda-12.1).

## 모델
| 모델 | 상태 | revision |
|---|---|---|
| Qwen2.5-0.5B-Instruct → `/data/models/qwen2.5_0.5b_instruct` | [V] 완료 (954 MB) | 7ae557604adf67be50417f59c2c2f167def9a775 |
| Llama-3.1-8B-Instruct | [V] 실패: GatedRepoError 403 (home0604 미승인) | main 0e9e39f2… (조회만) |

스크립트 `experiments/00_survey/download_models.py`, 로그 `/data/lsh/logs/phase1_download.log`, HF_HOME=/data/lsh/hf_cache.
