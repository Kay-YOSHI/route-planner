# Motivation
- 長期ツーリングに行く際，Google MyMapを使って事前に立ち寄りたいスポットや走るルートを引いたマップを作成しているが，下記の作業がまあまあめんどい
  - ルートを引く作業
    - 地点 *i* と *k* を繋ぐ作業をひたすら繰り返す必要がある
    - 基本下道を走りたいが，Google MyMapは是が非でも高速を使わせようとしてくる
  - 日を跨いだ距離の調整
    - 予め1日の走行距離の目安は決めているが，その距離を走った先が山奥とかだと宿が取れないため，N日目に距離を稼いで市街地まで行き，N+1日目は少し楽をするといった調整が必要
- 上記のような作業が楽になると嬉しい

# Objective
- 上記の作業を楽にするWebアプリを製作する

# Requirements
- 理想
  - 下記を入力すると，立ち寄りたいスポットや走りたい道をすべて通る<u>**全日程分の**</u>ルートを地図上に描画してくれるやつ（1日のゴールは宿がありそうな市街地にしてくれる）
    - スタート地点（＝ゴール地点＝自宅）
    - 立ち寄りたいスポットや走りたい道
    - 何日間の日程か
    - 1日の走行距離下限・上限
      - 何日間かと走行距離上限は良い感じにしないと整合性が取れなくなりそう（1日200km上限で5日間では帰ってこられない等）
- ミニマム
  - 一連の地点を入力するとそれらをすべて通過するルートを地図上に描画してくれるやつ
    - その後手動で良い感じにぶった切って全日程分のルートを完成させる

# Preliminaries
- 使えそうなツール
  - 巡回セールスマン問題の求解
    - google maps API（従量課金）
      - Directions API
        - リアルタイムの交通情報を備えた複数の交通手段のルート案内を提供する機能
      - Distance Matrix API
        - 複数経路の移動時間と距離を提供する機能
      - [Routes API](https://www.zenrin-datacom.net/solution/blog/gmpupdate-002)（上記2つのアップグレード版）
    - OR-Tools
      - 制約条件をいろいろ入れられそう
    - OSRM (Open Source Routing Machine)
      - 経路探索
      - 距離行列取得
  - 可視化
    - folium（Pythonで地図を可視化するためのライブラリ）
  - Webアプリ作成
    - streamlit

# Plan
## Step1
- 概要
  - 巡回セールスマン問題を解き，地図上にルートを可視化する（日帰りのイメージ）
- 利用ツール
  - 地点の緯度・経度取得
    - XXX
  - 巡回セールスマン問題を解く（ルートを算出）
    - OR-Tools
    - Google Distance Matrix API
  - 可視化
    - streamlit-folium
- 詳細ステップ
  - Step1. OR-Toolsを用いて巡回セールスマン問題を解く
    - Input: 
      - nodes: 巡回するN地点の[緯度, 経度]のリスト
        - どのように地点 *i* の緯度・経度を取得するか
          - Google Maps API
          - geocoder, geopy
      - dist: 距離行列（N×N行列．任意の2地点間の距離を格納）
        - ユークリッド距離（直線距離）
        - 車での移動距離（[Google Distance Matrix API](https://developers.google.com/optimization/routing/vrp?hl=ja#distance_matrix_api)で算出？）
    - Output: 
      - 総走行距離
      - 巡回する順序に並んだN地点のリスト
  - Step2. 可視化
    - Step1で得られた順序で各地点を巡るルートを描いてもらう
## Step2
- 概要
  - Step1を拡張？
    - 制約追加？
      - 1日の走行距離目安？複数日程？etc..
  - ChatGPTにツーリングスポットをリストアップしてもらう
    - ChatGPT API
    - 緯度経度も出せそうだが正確かどうかは不明（特にマイナーな場所）
  - foliumでストリートビューにアクセスできるようにする（[参考](https://www.youtube.com/watch?v=a9woRXmiy0s)）

# Memo
- 最適化問題を解けばよい？（巡回セールスマン問題的なアレ）
  - *min* 総走行距離とか総コスト（総ガソリン代？）とか
  - *s.t.*
    - 立ち寄りたい地点・走りたい道はすべて通る
    - 1日のゴール地点は市街地（宿がありそうな場所）
      - 例えば，ゴール地点の緯度経度が含まれる地域メッシュのラベルが「市街地」とか
        - 地域メッシュってラベル付いてんの？
    - 1日の走行距離<=1日の走行距離上限（あるいは幅を設けて1日の走行距離目安制約でもよい）
    - 高速は使わない
  - 決定変数は何すか？ルート？
- 最適化問題求解と地図をどう結び付ける？
  - そもそも最適化問題の（例えばダイクストラ法による）解は何か？（順序付けされた地点のリスト？）
  - であれば，経由する順序だけ最適化して（各地点をネットワークっぽく抽象化して），地点間のルートはgoogle先生に良い感じに引いてもらう？
- 1日ごとに最適化問題を解く？
  - 地点A → 地点B → 地点C → ・・・と走行距離を累積していって，1日の走行距離下限～上限の間に入ったらストップ，みたいな

# References
## General
- Python によるルート最適化の実践ガイド
  - https://qiita.com/haystacker/items/67dfc76fd35b65eccd89

## Google Maps API
- 【個人開発】Google Maps APIを利用して最適経路を提案するアプリ「Tabikochan」を作りました！
  - https://zenn.dev/lclco/articles/77d2af2e7bd24f

## OR-Tools
- 【〇】Google OR-Tools（公式）: 巡回セールスマン問題
  - https://developers.google.com/optimization/routing/tsp?hl=ja
- 【〇】第6回：OR-Toolsで巡回セールスマン問題を解く ～京都弾丸観光ツアーの作成を事例に～【ブレインパッドの数理最適化ブログ】
  - https://www.brainpad.co.jp/doors/contents/01_tech_2021-06-18-110000/
- 【〇】ortoolpy, tsp
  - 組合せ最適化 - 典型問題 - 巡回セールスマン問題
    - https://qiita.com/SaitoTsutomu/items/def581796ef079e85d02
  - 北海道内15箇所にポケモンマンホールが設置されるので最適経路を計算してみた
    - https://kiguchi999.hatenablog.com/entry/2019/12/08/%E5%8C%97%E6%B5%B7%E9%81%93%E5%86%8515%E7%AE%87%E6%89%80%E3%81%AB%E3%83%9D%E3%82%B1%E3%83%A2%E3%83%B3%E3%83%9E%E3%83%B3%E3%83%9B%E3%83%BC%E3%83%AB%E3%81%8C%E8%A8%AD%E7%BD%AE%E3%81%95%E3%82%8C
- Python+folium+openrouteserviceを使う (経路・時間行列・等時線を例に)
  - https://zenn.dev/takilog/articles/2be029ccd35972
- 【×】OR-Toolsで巡回セールスマン問題を解く
  - https://qiita.com/SaitoTsutomu/items/ab9d657c49879df69928
- 【×】OR-Toolsを使って巡回セールスマン問題を解き、効率的に沼津の聖地巡礼をする
  - https://jpn.pioneer/ja/piomatixlbsapi/blog/or-tools/
## Web Apps.
- OR-Tools + folium + streamlit
  - ルート最適化の結果を地図上に表示するアプリをstreamlit.ioにデプロイした件
    - https://zenn.dev/megane_otoko/articles/041_route_optimization
- streamlit-folium
  - 公式
    - https://folium.streamlit.app/
  - streamlitとfoliumを使用した都道府県別の新型コロナ感染者数ヒートマップを表示するデータアプリを作成する
    - https://zenn.dev/iwatagumi/articles/e6b57ff3730fb4
  - Streamlitでstreamlit-foliumを使って地図に情報を表示してみよう
    - https://welovepython.net/streamlit-folium/
## 緯度・経度取得
- Pythonと国土地理院APIで施設名から緯度経度を一括取得してCSV出力
  - https://zenn.dev/yamadamadamada/articles/3fb198003c5428