"""Simple Travelling Salesperson Problem (TSP) between cities."""

import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from sklearn.preprocessing import MinMaxScaler
import time

import streamlit as st
from streamlit_folium import st_folium
import folium  # Leaflet.js を使用した地理空間データの可視化
from folium import plugins  # folium ライブラリのための追加機能やプラグインを提供

import geocoder
import osmnx as ox  # OpenStreetMap のデータを取得、構築、分析し、ネットワーク分析を行う
import networkx as nx  # 複雑なネットワークの作成、操作、研究の用途
from ortools.constraint_solver import (
    pywrapcp,
)  # OR-Tools で、制約プログラミングをサポートする Python ラッパー
from ortools.constraint_solver import (
    routing_enums_pb2,
)  # OR-Tools で、ルーティング問題に関連する列挙型と定数を提供するモジュール


# map作成
def plot_map(
    dtf,
    y,
    x,
    start,
    zoom=12,
    tiles="openstreetmap",
    popup=None,
    size=None,
    color=None,
    legend=False,
    lst_colors=None,
):
    """
    folium でマップを作成する。
    :パラメータ
        param dtf: pandas
        param (y,x): str - (緯度，経度) を要素とする列
        param starting_point: list - (緯度，経度) を指定
        param tiles: str - "cartodbpositron", "OpenStreetMap", "Stamen Terrain", "Stamen Toner".
        param popup: str - クリックされたときにポップアップするテキストを指定する列
        :param size: str - サイズを変数で指定する列，None の場合は size=5 となる
        :param color: str - 変数 color を指定する列，None の場合はデフォルトの色になる
        :param lst_colors: list - 色の列が None でない場合に利用される、複数の色のリスト
        param marker: str - 変数 marker を指定する列，最大 7 個のユニークな値を取る
    :戻り値
        表示するマップオブジェクト
    """

    data = dtf.copy()

    ## プロットのためのカラムを作成
    if color is not None:
        lst_elements = sorted(list(dtf[color].unique()))
        lst_colors = (
            ["#%06X" % np.random.randint(0, 0xFFFFFF) for i in range(len(lst_elements))]
            if lst_colors is None
            else lst_colors
        )
        data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

    if size is not None:
        scaler = sp.MinMaxScaler(feature_range=(3, 15))
        data["size"] = scaler.fit_transform(data[size].values.reshape(-1, 1)).reshape(
            -1
        )

    ## マップ
    map_ = folium.Map(
        location=start,  # 地図の中心位置
        tiles=tiles,  # タイル
        zoom_start=zoom,  # ズームレベル
    )

    # if marker is None
    if (size is not None) and (color is None):
        data.apply(
            lambda row: folium.CircleMarker(
                location=[row[y], row[x]],
                popup=row[popup],
                color="#3186cc",
                fill=True,
                radius=row["size"],
            ).add_to(map_),
            axis=1,
        )
    elif (size is None) and (color is not None):
        data.apply(
            lambda row: folium.Marker(
                location=[row[y], row[x]],
                # popup=row[popup],
                tooltip=row[popup],
                # color=row["color"],
                icon=folium.Icon(color=row["color"], icon="map-marker"),
                fill=True,
                radius=10,
            ).add_to(map_),
            axis=1,
        )
    elif (size is not None) and (color is not None):
        data.apply(
            lambda row: folium.CircleMarker(
                location=[row[y], row[x]],
                popup=row[popup],
                color=row["color"],
                fill=True,
                radius=row["size"],
            ).add_to(map_),
            axis=1,
        )
    else:
        data.apply(
            lambda row: folium.CircleMarker(
                location=[row[y], row[x]],
                popup=row[popup],
                color="#3186cc",
                fill=True,
                radius=10,
            ).add_to(map_),
            axis=1,
        )

    ## tiles
    # タイルセットを追加
    layers = {
        "cartodbpositron": None,  # 組み込みレイヤー
        "openstreetmap": None,  # 組み込みレイヤー
        # "Stamen Terrain": "http://tile.stamen.com/terrain/{z}/{x}/{y}.jpg",
        # "Stamen Water Color": "http://tile.stamen.com/watercolor/{z}/{x}/{y}.jpg",
        # "Stamen Toner": "http://tile.stamen.com/toner/{z}/{x}/{y}.png",
        # "cartodbdark_matter": None,  # 組み込みレイヤー
    }

    # タイルレイヤーをマップに追加するループ
    for name, url in layers.items():
        if url:
            # カスタムタイルレイヤーを追加
            folium.TileLayer(
                tiles=url,
                attr="Map data © OpenStreetMap contributors, Stamen Design",
                name=name,
            ).add_to(map_)
        else:
            # 組み込みタイルレイヤーを追加
            folium.TileLayer(name).add_to(map_)

    ## 凡例
    if (color is not None) and (legend is True):
        legend_html = (
            """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""
            + color
            + """:</b><br>"""
        )
        for i in lst_elements:
            legend_html = (
                legend_html
                + """&nbsp;<i class="fa fa-circle fa-1x" style="color:"""
                + lst_colors[lst_elements.index(i)]
                + """"></i>&nbsp;"""
                + str(i)
                + """<br>"""
            )
        legend_html = legend_html + """</div>"""
        map_.get_root().html.add_child(folium.Element(legend_html))

    ## フルスクリーン
    plugins.Fullscreen(
        position="topright",
        title="展開",
        title_cancel="退出",
        force_separate_button=True,
    ).add_to(map_)

    return map_


def main(map_col, spot_list):

    # 時間計測開始
    # start_time = time.perf_counter()

    # データ生成
    # TODO: マイナー？な地名を入れるとTypeErrorが出る
    # --> TypeError: 'NoneType' object is not subscriptable
    dtf_list = []
    spot_counter = 0
    for i in range(len(spot_list)):
        # 緯度経度が取得できない地名が入力されたときは，その地点をskipする
        try:
            ret = geocoder.osm(spot_list[i], timeout=5.0)  # 緯度経度取得
            tmp = [spot_counter, spot_list[i], ret.latlng[0], ret.latlng[1]]
            dtf_list.append(tmp)
            spot_counter = spot_counter + 1
        except TypeError:
            pass
    print(dtf_list)

    cols = ["id", "Name", "y", "x"]  # y: 緯度, x: 経度
    dtf = pd.DataFrame(data=dtf_list, columns=cols)

    # 基点となる場所の設定
    i = 0  # id=0の地点を基点とする
    dtf["base"] = dtf["id"].apply(lambda x: 1 if x == i else 0)
    start = dtf[dtf["base"] == 1][["y", "x"]].values[0]
    # end = start  # 終点＝始点とする
    # print(f"From: {start} --> To: {end}")
    # print(dtf.head())

    # ネットワークグラフの作成
    # - 'start'の位置から半径10000メートル以内の道路ネットワークを取得
    # - network_type = "drive" or "bike" or "walk"
    G = ox.graph_from_point(start, dist=10000, network_type="drive")

    # ネットワークグラフの各エッジ（道路）に情報を追加
    G = ox.add_edge_speeds(G)  # 速度
    G = ox.add_edge_travel_times(G)  # 旅行時間

    # ネットワークグラフのプロット
    #   - 背景色: 黒
    #   - ノードのサイズ: 5
    #   - ノード色: 白
    #   - 図のサイズ: 幅16インチ、高さ8インチ
    # fig, ax = ox.plot_graph(
    #    G, bgcolor="black", node_size=5, node_color="white", figsize=(16, 8)
    # )

    # 道路ネットワークGから各地点に対応する（最も近い）ノードを取得
    dtf["node"] = dtf[["y", "x"]].apply(
        lambda x: ox.distance.nearest_nodes(G, x[1], x[0]), axis=1
    )
    # print(dtf.head())

    # 重複するノードを削除し、最初に出現するもののみを保持
    # dtf = dtf.drop_duplicates("node", keep='first')

    # 距離計算関数
    def f(a, b):
        """
        ノードaからbの最短経路の長さを計算
        """
        try:
            d = nx.shortest_path_length(
                G, source=a, target=b, method="dijkstra", weight="length"
            )
        except:
            d = np.nan
        return d

    # 距離行列を作成
    distance_matrix = np.asarray(
        [[f(a, b) for b in dtf["node"].tolist()] for a in dtf["node"].tolist()]
    )

    # 距離行列を Pandas データフレームに変換
    distance_matrix = pd.DataFrame(
        distance_matrix, columns=dtf["Name"].values, index=dtf["Name"].values
    )
    # print(distance_matrix.head())

    # 始点ノード，総巡回地点，ドライバー数
    start_node = dtf[dtf["id"] == i]["node"][0]  # 始点ノード
    lst_nodes = dtf["node"].tolist()  # ノードのリスト
    drivers = 1  # ドライバー数
    # print(
    #    f"始点ノード: {start_node}, 総巡回地点: {len(lst_nodes)-1}, ドライバー数: {drivers}"
    # )

    # Index Manager（nodeとindexの紐づけを管理してくれるもの）を作成
    #  - node: 我々が認識している地点番号
    #  - index: ソルバー内部で処理される際に用いられるもの
    manager = pywrapcp.RoutingIndexManager(
        len(lst_nodes),  # 地点数
        drivers,  # ドライバー数
        lst_nodes.index(start_node),  # 始点/終点のindex
    )

    # Routing Modelを作成
    model = pywrapcp.RoutingModel(manager)

    # 距離コールバック（距離取得関数，距離コスト評価関数）作成
    # from_nodeからto_nodeへの移動（距離）コストをdistance_matrixから見つけるコールバック
    # コールバック関数：ある関数を呼び出すときに引数に指定する別の関数
    def get_distance(from_index, to_index):
        """Returns the distance between the two nodes."""
        return distance_matrix.iloc[from_index, to_index]

    # 距離コールバックをRouting Modelに登録
    distance = model.RegisterTransitCallback(get_distance)

    # すべての車両に距離コスト評価関数を設定
    model.SetArcCostEvaluatorOfAllVehicles(distance)

    # MEMO: いろいろ設定（制約条件）を追加できるっぽい

    # デフォルトのルーティング検索パラメータを取得
    parameters = pywrapcp.DefaultRoutingSearchParameters()

    # 初期解を生成する戦略（first_solution_strategy）を設定
    # PATH_CHEAPEST_ARC:
    # ・最もコストの低いアーク（経路のセグメント）を選択する戦略
    # ・今いる地点から最短距離の地点にルートをつないでいくイメージ
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # 最適化実行
    solution = model.SolveWithParameters(parameters)

    # ルート（リスト）と総走行距離を取得
    def get_route_and_distance(solution, model, manager):
        # ドライバー#0 の開始インデックスを取得
        index = model.Start(0)
        # ルート（リスト），総走行距離
        route_idx, route_distance = [], 0
        # ルートの各ノードを走査
        while not model.IsEnd(index):
            # 現在のインデックスに対応するノードをリストに追加
            route_idx.append(manager.IndexToNode(index))
            previous_index = index
            # 次のインデックスを取得
            index = solution.Value(model.NextVar(index))

            # 距離を更新
            try:
                # distance_matrixから距離を取得し，ルートの総距離に加算
                route_distance += get_distance(previous_index, index)
            except:
                # エラーを吐いた場合は距離を取得
                route_distance += model.GetArcCostForVehicle(
                    from_index=previous_index, to_index=index, vehicle=0
                )
        route_idx.append(manager.IndexToNode(index))
        return route_idx, route_distance

    # 最適ルートと総走行距離を取得
    optimal_route, optimal_distance = get_route_and_distance(solution, model, manager)

    # optimal_routes（インデックスのリスト）をノードのリストに変換
    lst_route = [lst_nodes[i] for i in optimal_route]

    # ノード間の経路を取得する関数
    def get_path_between_nodes(lst_route):
        lst_paths = []
        for i in range(len(lst_route)):
            try:
                # 現在のノードと次のノードを取得
                a, b = lst_route[i], lst_route[i + 1]
            except:
                # 次のノードがない場合はループを終了
                break
            try:
                # グラフ G 上でノード a からノード b までの最短経路を計算
                # ダイクストラ法を使用し、重みは 'travel_time'（移動時間）で計算
                path = nx.shortest_path(
                    G, source=a, target=b, method="dijkstra", weight="travel_time"
                )
                # 経路が存在する場合（長さが 1 より大きい場合）、リストに追加
                if len(path) > 1:
                    lst_paths.append(path)
            except:
                # 経路が見つからない場合は次のループへ
                continue
        return lst_paths

    # ルート内の各ノード間の経路を取得
    lst_paths = get_path_between_nodes(lst_route)

    # 地図上にルートを描画
    map_ = plot_map(
        dtf,
        y="y",
        x="x",
        start=start,
        zoom=12,
        tiles="openstreetmap",
        popup="Name",
        color="base",
        lst_colors=["blue", "red"],
    )
    for path in lst_paths:
        ox.plot_route_folium(G, route=path, route_map=map_, color="red", weight=2)

    # 時間計測終了
    # end_time = time.perf_counter()

    # streamlit
    # TODO:optimal_routeを表示させるなら地点名が良いか（IndexToName？）
    with map_col:
        st.write(f"Total Distance: {round(optimal_distance/1000, 2)} km")  # 総走行距離
        st_folium(map_, width=1400, height=600, returned_objects=[])
        # st.write(f"Route: {optimal_route}) # 最適経路
        # st.write(f"Computation time: {'{:.2f}'.format((end_time-start_time)/60)} min.") # 計算時間


if __name__ == "__main__":

    # streamlit
    # to view Streamlit app on a browser, run it with the following command:
    # streamlit run C:\Users\ksk01\Git\route-planner\route_planner.py

    # streamlit: ページ設定
    st.set_page_config(page_title="Route-Planner", page_icon="🏍️", layout="wide")

    # タイトル
    st.title("Route-Planner")

    # 画面を分割
    placeholder = st.empty()
    menu_col, map_col = placeholder.columns([1, 4])

    with menu_col:

        # 入力フォーム
        if "list" not in st.session_state:
            st.session_state.list = [0]

        with st.form("form"):
            st_input_area = st.container()
            st_add_button_area = st.container()

            if st_add_button_area.form_submit_button(label="ADD FORM"):
                st.session_state.list.append(0)

            if st_add_button_area.form_submit_button(label="REMOVE FORM"):
                del st.session_state.list[-1]

            for i in range(len(st.session_state.list)):
                if i == 0:
                    st.session_state.list[i] = st_input_area.text_input(
                        f"Starting Point", key=i
                    )
                else:
                    st.session_state.list[i] = st_input_area.text_input(
                        f"Waypoint {i}", key=i
                    )

            is_calc = st.form_submit_button(label="COMPUTE")

        if is_calc:

            # 巡回する地点リスト
            spot_list = st.session_state.list
            # 実行
            main(map_col, spot_list)
