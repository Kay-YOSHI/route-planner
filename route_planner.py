"""Simple Travelling Salesperson Problem (TSP) between cities."""

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


# データ生成
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]  # 距離は整数
    data["num_vehicles"] = 1
    data["depot"] = 0  # ルートの始点と終点
    return data


# ルート（リスト）と総走行距離を取得
def get_route_and_distance(solution, routing, manager):
    # ドライバー#0 の開始インデックスを取得
    index = routing.Start(0)
    # ルート（リスト），総走行距離
    route_idx, route_distance = [], 0
    # ルートの各ノードを走査
    while not routing.IsEnd(index):
        # 現在のインデックスに対応するノードをリストに追加
        route_idx.append(manager.IndexToNode(index))
        previous_index = index
        # 次のインデックスを取得
        index = solution.Value(routing.NextVar(index))
        # 距離を取得し，ルートの総距離に加算
        route_distance += routing.GetArcCostForVehicle(
            from_index=previous_index, to_index=index, vehicle=0
        )
    return route_idx, route_distance


def main():
    # データ生成
    # TODO: データはpandas.dataFrameで持っとくか
    data = create_data_model()

    # Index Managerを作成
    # nodeとindexの紐づけを管理してくれるもの
    # node: 我々が認識している地点番号
    # index: ソルバー内部で処理される際に用いられるもの
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),  # 地点数
        data["num_vehicles"],  # 車両数
        data["depot"],  # 始点/終点のindex
    )

    # Routing Modelを作成
    routing = pywrapcp.RoutingModel(manager)

    # 距離コールバック（距離取得関数，距離コスト評価関数）作成
    # from_nodeからto_nodeへの移動（距離）コストをdistance_matrixから見つけるコールバック
    # コールバック関数：ある関数を呼び出すときに引数に指定する別の関数
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    # 距離コールバックをRouting Modelに登録
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # すべての車両に距離コスト評価関数を設定
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # MEMO: いろいろ設定（制約条件）を追加できるっぽい

    # デフォルトのルーティング検索パラメータを取得
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    # 初期解を生成する戦略（first_solution_strategy）を設定
    # PATH_CHEAPEST_ARC:
    # ・最もコストの低いアーク（経路のセグメント）を選択する戦略
    # ・今いる地点から最短距離の地点にルートをつないでいくイメージ
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # 最適化実行
    solution = routing.SolveWithParameters(search_parameters)

    # 最適ルートと総走行距離を取得
    optimal_route, optimal_distance = get_route_and_distance(solution, routing, manager)
    print(f"Route: {optimal_route}")
    print(f"Total Distance: {optimal_distance} miles")

    # TODO:
    # optimal_routes（インデックスのリスト）をノードのリストに変換
    # 任意の2ノード間の最短経路を算出し，地図上にルートを描画


if __name__ == "__main__":
    main()
