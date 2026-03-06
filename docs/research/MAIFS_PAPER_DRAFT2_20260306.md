# DAAC: 다중 에이전트 이미지 위변조 탐지를 위한 불일치 인식 적응형 합의 기법

---

## 요약

생성형 AI 확산으로 이미지 위변조는 편집 조작에서 Diffusion·GAN 기반 합성까지 다양해졌고, 단일 탐지기는 도메인 편향을 피하기 어렵다. 다중 전문 에이전트를 조합하는 접근이 해법이나, 기존 trust 기반 규칙형 합의(COBRA)는 에이전트 간 구조적 불일치 패턴을 학습하지 못한다는 한계가 있다. 본 논문은 에이전트 간 불일치를 43차원 메타 특징으로 학습하는 **DAAC(Disagreement-Aware Adaptive Consensus)** 기법을 제안한다. 4개 전문 에이전트, 1,500장, 10-seed 반복 평가에서 DAAC-GBM은 macro-F1 0.861로 COBRA(0.266) 대비 유의하게 우수하며(p=0.00195), 6개 데이터 조합 60회 반복에서도 일관된 우위를 확인했다(sign 60/0, p=1.63e-11).

---

## 1. 서론

이미지 위변조 탐지는 JPEG 압축 흔적, 센서 노이즈 불일치, 공간적 조작 흔적, 생성 모델 아티팩트 등 이질적 단서를 동시에 처리해야 한다. 단일 모델의 도메인 편향은 오탐·미탐으로 직결되므로, 서로 다른 탐지 원리를 가진 다수의 전문 에이전트를 조합하는 다중 에이전트 접근이 자연스러운 해법이다.

그러나 **합의(consensus) 단계가 병목**이 된다. 기존 trust 기반 규칙형 합의(COBRA 계열: RoT/DRWA/AVGA)는 사전 고정된 trust 점수와 handcrafted 수식으로 집계를 수행한다. 이는 에이전트들이 특정 클래스에 대해 가지는 구조적 맹점—AI 생성 이미지를 감지 못하는 주파수 분석 에이전트, 조작 탐지 능력이 없는 생성 탐지 에이전트—의 패턴을 데이터로부터 학습하지 못한다.

본 논문은 **DAAC** 기법을 제안한다. 핵심 아이디어는 "누가 맞았는가"가 아닌 **"에이전트들이 어떻게 서로 다르게 틀리는가"** 를 학습 신호로 활용하는 것이다. 에이전트 간 pairwise 충돌로 구성된 43차원 메타 특징 위에 GBM 분류기를 학습하면, 개별 에이전트의 클래스별 맹점이 오히려 풍부한 구분 신호로 전환된다.

---

## 2. 관련 연구

COBRA[1]는 에이전트별 trust 점수 기반 합의를 제공하며 RoT/DRWA/AVGA 세 변형을 포함한다. DRWA는 샘플별 분산으로 가중치를 조정하고 AVGA는 softmax temperature를 조정하지만, 모두 사전 고정된 trust 점수를 출발점으로 삼는 handcrafted 규칙이다. oracle trust를 사용해도 F1이 오히려 소폭 하락하여 합의 방식 자체의 표현력 한계가 지배적임을 확인한다. 이미지 포렌식 분야에서는 FatFormer[2], TruFor[3] 등이 제안되었으나, 다중 에이전트 합의 단계에서 불일치 패턴을 학습 신호로 활용한 연구는 드물다.

---

## 3. DAAC 기법

### 3.1 문제 정의

$n$개의 에이전트가 이미지 $x$에 대해 판정 $v_i \in \mathcal{V}$와 confidence $c_i \in [0,1]$을 독립 산출한다. 여기서 $\mathcal{V} = \{$authentic, manipulated, ai\_generated, uncertain$\}$은 에이전트 수준의 판정 집합이다.
 DAAC의 학습 목표는 최종 클래스 $\mathcal{C} = \{$authentic, manipulated, ai\_generated$\}$에 대한 합의 함수 $f: \{(v_i, c_i)\}_{i=1}^n \to \mathcal{C}$를 데이터로부터 학습하는 것이다. COBRA는 $f$를 handcrafted 수식으로 정의하지만, DAAC는 불일치 구조를 포함한 43차원 메타 특징 위에서 $f$를 학습한다.

### 3.2 43차원 메타 특징 설계

| 그룹 | 내용 | 차원 |
|------|------|---:|
| Per-agent | verdict one-hot (4×4) + raw confidence | 20 |
| Pairwise disagreement | $\mathbf{1}[v_i \neq v_j]$ + $\|c_i - c_j\|$ + $(c_i+c_j)\cdot\mathbf{1}[v_i \neq v_j]$ (6쌍×3) | 18 |
| Aggregate | confidence variance, verdict entropy, max-min gap, unique verdict count, majority ratio | 5 |
| **합계** | | **43** |

**<표 1> DAAC 43차원 메타 특징 구성**

### 3.3 에이전트 구성

| 에이전트 | 백엔드 | 분석 원리 | 주요 맹점 |
|----------|--------|----------|----------|
| Frequency | CAT-Net | JPEG 압축 흔적 탐지 | AI-generated F1=0.000 |
| Noise | MVSS-Net | PRNU/SRM 센서 노이즈 불일치 | AI-generated F1=0.000 |
| FatFormer | CLIP ViT-L/14 + DWT | AI 생성 이미지 탐지 | Manipulated F1=0.000 |
| Spatial | Mesorch | ViT 기반 부분 조작 영역 탐지 | AI-generated F1=0.000 |

**<표 2> MAIFS 4개 전문 에이전트 구성**

DAAC의 제안은 에이전트 자체가 아닌 **합의 계층**에 있다. 에이전트 맹점은 서로 상보적이며, DAAC는 이 충돌 구조를 학습 신호로 활용한다.

---

## 4. 실험 설정

### 4.1 데이터 및 프로토콜

| 항목 | Protocol-P (주 실험) | Protocol-M (일반화 검증) |
|------|---------------------|------------------------|
| 목적 | 방법 간 성능 비교 | 데이터셋 이동에 대한 일반화 검증 |
| 데이터셋 | CASIA2 (Au 500 + Tp 500) + GenImage BigGAN (500) | DS-A~D (출처·유형 교차 4종) + OpenSDI + AI-GenBench proxy |
| 총 샘플 | 1,500장 | 조합별 상이 |
| Split | train/val/test = 0.6/0.2/0.2 | train/val/test = 0.6/0.2/0.2 |
| 반복 횟수 | seeds 300~309 (10회) | 6개 조합 × 10 seeds = 60 runs |
| 통계 검정 | paired Wilcoxon vs COBRA | sign test + Wilcoxon |

**<표 7> 실험 프로토콜 비교**

표 7은 두 실험 프로토콜의 목적·데이터셋·반복 설정을 비교한 것이다.

### 4.2 비교 방법

규칙형 기준선으로 Majority Vote, Weighted MV, COBRA(drwa)를 사용하며, COBRA(drwa)를 주 baseline으로 삼는다. 학습형 방법으로는 A4 특징(20-dim)에 LR을 적용한 Logistic Stacking과, 제안 방법인 DAAC(A5 43-dim 기반 LR/GBM/MLP)를 비교한다. COBRA는 RoT/DRWA/AVGA 세 변형을 포함하는 trust 기반 합의의 대표 방법으로, 동일한 다중 에이전트 설정에서 비교 가능한 다른 공개 합의 알고리즘이 존재하지 않아 주 baseline으로 채택하였다.

---

## 5. 실험 결과

### 5.1 주요 성능 비교 (Protocol-P, 10 seeds)

| 방법 | Macro-F1 (mean±std) | p vs COBRA |
|------|--------------------:|----------:|
| Majority Vote | 0.2774±0.0178 | 0.0371 |
| Weighted MV | 0.2664±0.0128 | 1.0000 |
| **COBRA (drwa)** | **0.2664±0.0090** | — (기준) |
| Logistic Stacking (A4+LR) | 0.8500±0.0161 | 0.00195** |
| DAAC-LR (A5+LR) | 0.8541±0.0087 | 0.00195** |
| **DAAC-GBM (A5+GBM)** | **0.8613±0.0156** | **0.00195**** |
| DAAC-MLP (A5+MLP) | 0.8570±0.0122 | 0.00195** |

**<표 3> 방법 간 Macro-F1 비교 (Protocol-P, 10 seeds)**

** p<0.01, paired Wilcoxon. DAAC 계열은 모든 10 seed에서 COBRA를 능가한다.

### 5.2 클래스별 성능 및 에이전트 맹점

| 방법 | Authentic F1 | Manipulated F1 | AI-generated F1 |
|------|------------:|---------------:|----------------:|
| Frequency (단독) | 0.804 | 0.481 | **0.000** |
| Noise (단독) | 0.558 | 0.476 | **0.000** |
| FatFormer (단독) | 0.542 | **0.000** | 0.473 |
| Spatial (단독) | 0.570 | 0.355 | **0.000** |
| COBRA (drwa) | 0.573 | 0.199 | 0.027 |
| **DAAC-GBM** | **0.821** | **0.800** | **0.963** |

**<표 4> 클래스별 F1 및 에이전트 맹점 (10-seed 평균)**

Frequency/Noise/Spatial은 ai\_generated F1=0.000, FatFormer는 manipulated F1=0.000의 구조적 맹점을 가진다. COBRA는 이 맹점을 가중 합산으로 전파하는 반면, DAAC-GBM은 에이전트 간 충돌 패턴을 학습하여 세 클래스 모두 균형 잡힌 성능을 달성한다. Cohen's Kappa: COBRA κ=0.068 (우연 수준) vs DAAC-GBM κ=0.796 (강한 일치).

### 5.3 특징 Ablation (GBM, 10 seeds)

| 구성 | 차원 | Macro-F1 |
|------|-----:|---------:|
| A1: confidence only | 4 | 0.7972±0.0149 |
| A2: verdict only | 16 | 0.7596±0.0213 |
| A3: disagreement only | 23 | 0.8534±0.0102 |
| A4: verdict+confidence | 20 | 0.8506±0.0202 |
| **A5: full DAAC (제안)** | **43** | **0.8613±0.0156** |

**<표 5> 특징 그룹 Ablation 결과 (GBM, 10 seeds)**

A3(불일치만, 23-dim) > A4(verdict+confidence, 20-dim)는 개별 판정·confidence보다 **불일치 패턴 자체**가 더 강한 클래스 구분 신호임을 의미한다. 최상위 특징 중요도는 `disagree_frequency_fatformer` (56.5%)로, FatFormer와 나머지 에이전트의 판정 충돌이 핵심 신호다.

### 5.4 다중 데이터 일반화 (Protocol-M, 60 runs)

6개 조합 전체에서 DAAC가 COBRA를 능가했다(sign 60/0, Wilcoxon p=1.63e-11, mean delta +0.493). 단 한 번도 역전되지 않아 특정 데이터 조합에 의한 우연 효과가 아님을 확인한다.

| 조합 | Validation Aspect | COBRA F1 | DAAC F1 | Delta |
|------|------------------|----------:|--------:|------:|
| DS-A (CASIA+BigGAN) | Baseline combination | 0.277 | 0.835 | +0.558 |
| DS-B (ImageNet+CASIA+BigGAN) | Authentic source shift | 0.274 | 0.867 | +0.593 |
| DS-C (CASIA+IMD2020+BigGAN) | Manipulation type shift | 0.303 | 0.898 | +0.595 |
| DS-D (ImageNet+IMD2020+BigGAN) | Both sources unseen | 0.300 | 0.863 | +0.564 |
| OpenSDI | Real-world distribution | 0.177 | 0.409 | +0.232 |
| AI-GenBench proxy | Non-BigGAN generators | 0.333 | 0.750 | +0.417 |

**<표 6> 다중 데이터 조합별 일반화 결과 (Protocol-M)**

---

## 6. 결론

본 논문은 다중 에이전트 이미지 포렌식에서 에이전트 간 불일치 패턴을 명시적으로 학습하는 합의 기법 DAAC를 제안했다. DAAC-GBM은 COBRA 대비 macro-F1 0.861 vs 0.266(p=0.00195)의 유의한 우위를 보이며, 6개 데이터 조합 60회 반복에서도 일관된 우위를 확인했다(sign 60/0). 에이전트 성능 개선 없이 합의 계층만 개선해도 +0.43~+0.60의 F1 향상이 가능하며, 이는 다중 에이전트 시스템에서 합의 방식 설계가 개별 에이전트 성능만큼 중요함을 시사한다. 향후 연구로 FatFormer를 OpenSDI(SD/Flux) 데이터로 재학습하여 최신 Diffusion 이미지에 대한 성능 향상을 검토할 예정이다.

---

## 참고문헌

[1] Z. Haider, et al. "Consensus-Based Reward Model for Better Alignment of Large Language Models" Scientific Reports, Vol. 15, Article 4004, 2025.

[2] H. Liu, et al. "Learning to Detect AI-Generated Images via FatFormer" CVPR, Seattle, 2024, pp. 10770-10780.

[3] F. Guillaro, et al. "TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization" CVPR, Vancouver, 2023, pp. 20606-20615.
