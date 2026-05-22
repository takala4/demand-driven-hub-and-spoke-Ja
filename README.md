# 需要主導型の輸送ネットワーク形成： Hub and Spoke構造の創発

## 概要

本リポジトリでは，下記論文における数値計算コードを掲載している．

* 酒井 高良, 高山 雄貴: 需要主導型の輸送ネットワーク形成：Hub and Spoke構造の創発, 土木学会論文集, Vol. 81, No. 6, 24-00167, [https://www.jstage.jst.go.jp/article/jscejj/81/6/81_24-00167/_article/-char/ja/](https://www.jstage.jst.go.jp/article/jscejj/81/6/81_24-00167/_article/-char/ja/)
* 酒井 高良, 高山 雄貴: 需要主導型の輸送ネットワーク形成：Hub and Spoke構造の創発 (preprint), 2024, [https://doi.org/10.51094/jxiv.749](https://doi.org/10.51094/jxiv.749)


## 二つの実装

本リポジトリには，同一問題に対する 2 つの実装を含めている．両者は論文掲載 6 ケース全てで `opt_Z` と `SP_tree` が完全一致することを確認済み．

| | ファイル | 説明 |
|---|---|---|
| **v1** | `code/hubspoke_v1.py` | 論文通りの実装．アルゴリズムをそのまま素直に書き下したもの．アルゴリズム検証のベースライン |
| **v2** | `code/hubspoke.py` | 改良版．分枝限定法に以下の最適化を追加して大幅高速化（最大 34 倍） |

v2 に加えた最適化（いずれも結果を変えない刈り取り）：

* **skip-on-push**: 子の下界 `app_Z` が既に既知の上界 `opt_Z` 以上であれば heap に push しない
* **periodic heap rebuild**: `opt_Z` 改善時に heap 内の dead entry を一括削除
* **diff-based heap encoding**: heap entry を親への差分参照で表現しメモリを大幅圧縮
* **exact-leaf skip**: 線形緩和が exact (`Z ≈ app_Z`) な leaf は branching しても改善不可能なので push しない

これに加え，per-leaf solve を Numba JIT 化（dense Dijkstra + 線形化 + 流量伝播 + 下界・上界計算を 1 パスに融合）している．


## リポジトリ構成

```
code/
├── hubspoke.py            # v2: 改良版（Numba JIT + 分枝限定法のメモリ・刈り取り最適化）
├── hubspoke_v1.py         # v1: 論文通りの実装（参照用，アルゴリズム検証のベースライン）
├── reproduce_paper.ipynb  # 論文掲載 6 ケースを一括再現する notebook
├── run.ipynb              # 単一ケース実行例（既存）
├── benchmark.py           # v2 vs v1 の性能比較
├── profile_run.py         # cProfile によるプロファイル
├── verify_small.py        # 小規模ケースでの実装間照合（subprocess 隔離 + メモリ上限）
└── _verify_worker.py      # verify_small.py のワーカ
```


## 使用方法

### 依存ライブラリ

`numpy`, `pandas`, `networkx`, `matplotlib`, `numba`, `psutil`（`verify_small.py` のみ）．


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


### v2 と v1 の性能比較

```
cd code
python benchmark.py all              # 6 ケース全部
python benchmark.py "(e)"            # case (e) のみ
python benchmark.py all --no-v1      # v1 をスキップ（高速）
```


### 実装の正しさ検証（開発用）

`hubspoke.py` (v2) に変更を加えた後，小規模ケースで v1 と結果が一致するか確認できる：

```
cd code
python verify_small.py 5              # N=5, v2 vs v1
python verify_small.py 10 --phi 50 --rho 2.5
```

将来 `hubspoke_<variant>.py` のような派生実装を追加する場合は `--impl hubspoke_<variant>` で対象を指定できる．


## 連絡先

東京科学大学　[酒井高良](https://takala4.github.io/cv/)（sakai.t.dcad@m.isct.ac.jp）
