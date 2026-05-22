# 需要主導型の輸送ネットワーク形成： Hub and Spoke構造の創発

## 概要

本リポジトリでは，下記論文における数値計算コードを掲載している．

* 酒井 高良, 高山 雄貴: 需要主導型の輸送ネットワーク形成：Hub and Spoke構造の創発, 土木学会論文集, Vol. 81, No. 6, 24-00167, [https://www.jstage.jst.go.jp/article/jscejj/81/6/81_24-00167/_article/-char/ja/](https://www.jstage.jst.go.jp/article/jscejj/81/6/81_24-00167/_article/-char/ja/)
* 酒井 高良, 高山 雄貴: 需要主導型の輸送ネットワーク形成：Hub and Spoke構造の創発 (preprint), 2024, [https://doi.org/10.51094/jxiv.749](https://doi.org/10.51094/jxiv.749)


## 五つの実装

本リポジトリには同一問題に対する 5 つの実装を含めている．論文掲載 6 ケース (N=17) で **v1, v2, v5 は厳密に opt_Z と SP_tree が一致**することを確認済み．v3 は確率的ヒューリスティック，v4 は決定的 ε-最適である（後述）．

| | ファイル | 種別 | 説明 |
|---|---|---|---|
| **v1** | `code/hubspoke_v1.py` | 厳密 (paper-faithful) | 論文の Soland 型分枝限定法をそのまま実装．アルゴリズム検証のベースライン |
| **v2** | `code/hubspoke.py` | **厳密 (improved)** | v1 と同じ BB を Numba JIT 化 + 探索フィルタ追加で大幅高速化 |
| **v3** | `code/hubspoke_v3.py` | **確率的ヒューリスティック** | Variable Neighborhood Search．大域収束保証なし．大規模 N で v2 が破綻する領域で実用解を秒オーダで取得 |
| **v4** | `code/hubspoke_v4.py` | 決定的 ε-最適（Gurobi 必要） | 凹費用を piecewise linear で近似し MILP として Gurobi で解く．流量が ν の整数倍に限られるので breakpoints を {0, ν, 2ν, ...} に置けば PWL は exact |
| **v5** | `code/hubspoke_v5.py` | **DP（仮説依存・条件付き厳密）** | 「最適な根付き木の部分木は連続区間」という経験的仮説に基づく DP．論文 6 ケースおよび `ρ/t` が 1 から離れている領域では v2 と完全一致．**ρ/t ≈ 1 かつ φ ≈ 0 の境界では仮説が崩れ得るので注意**．**論文 6 ケースをマイクロ秒で解く**．N=100 でも 220 ms |

### v2: 改良された分枝限定法（本命）

v1 と完全に同じアルゴリズム（Soland 型 spatial BB）を以下で高速化：

* **Numba JIT 化**：per-leaf solve（dense Dijkstra + 線形化 + 流量伝播 + 下界計算）を 1 パスに融合
* **skip-on-push**：子の下界 `app_Z ≥ opt_Z` の leaf は heap に push しない
* **periodic heap rebuild**：`opt_Z` 改善時に heap 内 dead entry を一括削除
* **diff-based heap encoding**：heap entry を親への差分参照で表現してメモリ大幅圧縮
* **exact-leaf skip**：線形緩和が exact (`Z ≈ app_Z`) な leaf は push しない（degenerate ループ回避）

最遅 case (d) で legacy v1 の 1 時間 7 分 → **94 秒（v2）**．論文 6 ケースを全て厳密証明可能．

### v3: VNS（確率的ヒューリスティック）

* 解 = ノード 0 を根とする有向木（各ノードの親配列）
* 1-opt 局所探索 + shake (k 個親同時変更) + random restart
* **大域収束保証**：`restarts > 0` のとき Brimberg-Mladenović (1996) の漸近収束．時間 → ∞ で確率 1 で大域最適到達．**有限時間保証なし**
* 論文 6 ケース全てを 10〜30 秒で論文値発見（**証明はしない**）．N=50 (L=2401) でも 30 秒で良質解
* 厳密証明が要らない探索フェーズや大規模実験に有用

### v5: 区間構造 DP（仮説依存の高速解法）

**動作原理**：「最適な根付き木のすべての部分木がノード番号の連続区間をなす」という経験的仮説に基づき，2 段の DP で解く：

- `f(a, b, r)`：区間 `[a, b]` を hub `r ∈ [a, b]` に集約する最小コスト
- `M(a, c, p)`：区間 `[a, c]` を複数の部分区間に分割し，各 hub が親 `p` に接続する最小コスト
- `f(a, b, r) = M(a, r-1, r) + M(r+1, b, r)`
- 最上位：`opt_Z = M(1, N-1, 0)`

**計算量**：時間 O(N⁵)，メモリ O(N³)．N=17 でマイクロ秒，N=100 で 220ms．

**論文 6 ケース再現**：6 ケース全て v2 と完全一致．case (d) は v2 の 94 秒 → v5 では **< 1ms**．

**仮説の妥当領域（重要）**：

- 仮説は **論文 6 ケースおよびランダム 50 ケース** （いずれも論文の仮定 2.1〜2.3 を満たす）で成立を実証．
- しかし論文の仮定 2.1〜2.3 だけでは仮説は保証されない．具体的に **`ρ/t` が 1 にきわめて近く，かつ `φ` が極めて小さい** 領域（例：`φ=0.1, ρ=1.01, t=1`）では，最適解が非区間部分木を持ち，v5 は数 % 〜 数十 % 悪い解を返すことがある．
- 論文 6 ケースは全て `ρ/t ≥ 2.5` かつ `φ ≥ 10` で，**安全領域** に十分入っている．
- 仮説が成立する厳密な条件の特徴付けは今後の課題．

**実務上の指針**：
- 論文の典型パラメータ範囲（`ρ/t ≥ 2`, `φ` がほどほどに大きい）では v5 を厳密解として使ってよい．
- `ρ/t` を 1 に近づけて実験する／`φ` を 0 に近づけて実験する場合は，**v2 とクロスチェック**してから v5 を使うこと．


### v4: Gurobi MILP + PWL（決定的 ε-最適）

* 流量は ν の整数倍に限られる性質を利用し，breakpoints を `{0, ν, 2ν, ..., (N-1)ν}` に置くことで **PWL 近似は exact**
* Gurobi が分枝限定 + cuts + presolve で解く
* 論文 6 ケースのうち (a)(b)(e) は **数秒で証明完了**．(c)(f) は数十秒〜数分で 0.4-0.6% 範囲．**(d) は LP 緩和が loose で短時間に証明不可**（5 分で gap 7%）
* warm-start サポート（v3 解を渡すと incumbent として再利用）

### 実行時間の比較 (N=17, 論文 6 ケース)

実測値（同一マシンでの 1 回計測．v3 は確率的なので参考値）．**太字 = 厳密証明完了．それ以外は注釈参照**．

| ケース | v1 | v2 | v3 (VNS) | v4 (Gurobi) | **v5 (DP)** |
|:-:|---:|---:|---:|---:|---:|
| (a) | **< 1 s** | **0.7 s** | < 1 s †‡ | **0.4 s** | **<1 ms** ‖ |
| (b) | **< 1 s** | **1.3 s** | 30 s †‡ | **1.1 s** | **<1 ms** ‖ |
| (c) | **4 分 48 秒** | **10.5 s** | 10 s †‡ | 60 s (0.4% gap)§ | **<1 ms** ‖ |
| (d) | **1 時間 8 分** | **118 s** | 10 s †‡ | 120 s (24% gap)§ | **<1 ms** ‖ |
| (e) | **29.7 s** | **2.3 s** | < 10 s †‡ | **4.9 s** | **<1 ms** ‖ |
| (f) | **16 分 44 秒** | **30.3 s** | < 10 s †‡ | 60 s (0.6% gap)§ | **<1 ms** ‖ |

† v3 は確率的ヒューリスティックなので大域最適性の証明はしない．これらの時間は **「論文 opt_Z に一致する解を発見した時間」** で，「証明された時間」ではない．seed や config で変動．
‡ 厳密一致は確認済み（論文値と `opt_Z` および `SP_tree` がビット単位で一致）．
§ v4 の "gap" は Gurobi MIPGap．time-limit 到達で打ち切り．**解は paper opt_Z に一致しているが LP 緩和側の closing が間に合わず証明未達**．時間を伸ばせば最終的に証明可能．
‖ v5 は経験的仮説「部分木 = 連続区間」を利用．論文 6 ケース（`ρ/t ≥ 2.5, φ ≥ 10`）で v2 と完全一致．`ρ/t ≈ 1` かつ `φ ≈ 0` の境界では仮説が崩れる場合あり（前述）．Numba JIT 初回コンパイル ~0.5s は除外．


### スケーラビリティ（case (d) 型，phi=10, rho=5）

| 問題規模 | v2 (BB) | v3 (VNS) | v4 (Gurobi) | **v5 (DP)** |
|---|---|---|---|---|
| N=17 | 94 s で証明 | 10 s で発見 | 5 分で gap 7.6% | **<1 ms 証明 (仮説下)** |
| N=25 | 困難 | 20 s で解 | timeout 推定 | **<1 ms** |
| N=50 | 不可能 | 34 s で解 | timeout 推定 | **10 ms** |
| N=100 | 不可能 | 数分で解（証明なし） | 未確認 | **220 ms** |


### 使い分け（要約）

| 用途 | 推奨 |
|---|---|
| **論文の数値実験を再現したい** | **v5** が圧倒的高速（μs〜ms） |
| **`ρ/t ≈ 1` かつ `φ ≈ 0` の境界パラメータでの解析** | v5 ではなく **v2** を使う（v5 は仮説違反域に入り得る） |
| **v5 のアルゴリズムを別実装と照合したい** | v2（小規模で v1 も併用） |
| 大規模 N で良質解（証明不要） | **v3 (VNS)** |
| 一般的な MILP 解法を使いたい | v4 (Gurobi) |


## リポジトリ構成

```
code/
├── hubspoke.py            # v2: 改良版 BB（本命，Numba JIT + 探索フィルタ）
├── hubspoke_v1.py         # v1: 論文通りの BB（参照用，アルゴリズム検証のベースライン）
├── hubspoke_v3.py         # v3: VNS 確率的ヒューリスティック
├── hubspoke_v4.py         # v4: Gurobi MILP + PWL 決定的 ε-最適
├── hubspoke_v5.py         # v5: 区間 DP（論文の制約下で μs〜ms オーダの厳密解）
├── v5_NOTES.md            # v5 の理論的考察メモ（仮説・反例・未解決問題）
├── viz.ipynb              # 論文掲載用の図を生成する notebook（ハードコード SP_tree）
├── viz_utils.py           # 紙質の link_plot ユーティリティ（任意の SP_tree を可視化）
├── reproduce_paper.ipynb  # 論文掲載 6 ケースを一括再現する notebook
├── demo_refine.ipynb      # 空間細分化実験：case (d) で v5 を解いて構造の収束を観察（v5 スケーラビリティ）
├── run.ipynb              # 単一ケース実行例（既存）
├── benchmark.py           # 各実装の性能比較
├── profile_run.py         # cProfile によるプロファイル
├── verify_small.py        # 小規模ケースでの実装間照合（subprocess 隔離 + メモリ上限）
└── _verify_worker.py      # verify_small.py のワーカ
```


## 使用方法

### 依存ライブラリ

* 必須: `numpy`, `pandas`, `networkx`, `matplotlib`, `numba`
* v3 と verify ハーネス: `psutil`
* v4: `gurobipy`（Gurobi ライセンスが必要．アカデミック無償ライセンス可）


### 論文の数値実験を再現

```
cd code
jupyter notebook reproduce_paper.ipynb
```

全 6 ケース (a)〜(f) を v2 で順に解き，論文掲載値との照合とネットワーク構造の可視化を行う．


### 単一ケースの実行

```
cd code
jupyter notebook run.ipynb
```


### v1〜v4 の性能比較

```
cd code
python benchmark.py all              # 全 6 ケース，v1/v2 比較
python benchmark.py "(e)"            # case (e) のみ
python benchmark.py all --no-v1      # v1 をスキップ（高速）
```


### 実装の正しさ検証（開発用）

実装に変更を加えた後，小規模ケースで結果が一致するか確認できる：

```
cd code
python verify_small.py 5                          # N=5, v2 vs v1（default）
python verify_small.py 10 --phi 50 --rho 2.5      # case (e) パラメータ
python verify_small.py 17 --impl v3 --phi 10 --rho 5  # v3 単独
python verify_small.py 17 --impl both --baseline v1 --impl v3  # 任意実装ペア比較
```

将来 `hubspoke_v5.py` 等の派生実装を追加する場合は `--impl v5` または `--impl hubspoke_<variant>` で対象を指定できる．worker の `IMPL_MODULES` 辞書にエイリアスを追加すると短縮形が使える．


## 連絡先

東京科学大学　[酒井高良](https://takala4.github.io/cv/)（sakai.t.dcad@m.isct.ac.jp）
