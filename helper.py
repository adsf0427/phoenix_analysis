import ctypes
import json
import sys
# 加载共享库
lib = ctypes.CDLL('./libanalysis.so')

# 定义函数参数和返回类型
lib.AnalysisHumanTiles.argtypes = [ctypes.c_char_p]
lib.AnalysisHumanTiles.restype = ctypes.c_char_p
lib.FreeString.argtypes = [ctypes.c_char_p]

def analysis_human_tiles(human_tiles_info):
    # 将 Python 字典转换为 JSON 字符串
    input_json = json.dumps(human_tiles_info).encode('utf-8')
    
    # 调用 Go 函数
    result = lib.AnalysisHumanTiles(input_json)
    
    # 将结果转换回 Python 对象
    output = json.loads(result.decode('utf-8'))
    
    # 释放 C 字符串
    # lib.FreeString(result)
    
    return output

# 定义函数参数和返回类型
lib.AnalysisPlayerWithRisk.argtypes = [ctypes.c_char_p]
lib.AnalysisPlayerWithRisk.restype = ctypes.c_char_p

def analysis_player_with_risk(player_info):
    # 将 Python 字典转换为 JSON 字符串
    input_json = json.dumps(player_info).encode('utf-8')
    
    # 调用 Go 函数
    result = lib.AnalysisPlayerWithRisk(input_json)
    
    # 将结果转换回 Python 对象
    output = json.loads(result.decode('utf-8'))
    
    return output

def analysis_tehai(tehai_str, other_info):
    human_tiles_info = {
        "HumanTiles": tehai_str,
        "HumanMelds": [],
        "HumanDoraTiles": "",
        "HumanTargetTile": "",
        "IsTsumo": False
    }
    player_info = analysis_human_tiles(human_tiles_info)
    
    player_info['DoraTiles'] = other_info['DoraTiles']
    player_info['RoundWindTile'] = other_info['RoundWindTile']
    player_info['SelfWindTile'] = other_info['SelfWindTile']
    player_info['IsParent'] = other_info['IsParent']
    player_info['DiscardTiles'] = other_info['DiscardTiles']
    player_info['LeftTiles34'] = other_info['LeftTiles34']
    player_info['LeftDrawTilesCount'] = other_info['LeftDrawTilesCount']
    
    # print(f"player_info: {player_info}")
    
    result = analysis_player_with_risk(player_info)

    return result

if __name__ == "__main__":
    human_tiles_info = {
        "HumanTiles": "234688m 34s # 0555P 234p",
        "HumanMelds": [],
        "HumanDoraTiles": "",
        "HumanTargetTile": "",
        "IsTsumo": False
    }

    result = analysis_human_tiles(human_tiles_info)
    print(result)
    # sys.exit(0)

    player_info = {
        "HandTiles34": [0, 0, 1, 1, 1, 1, 0, 1, 0, 
                        0, 0, 0, 0, 1, 1, 1, 1, 0, 
                        0, 1, 1, 0, 1, 1, 1, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0],
        "Melds": [],
        "DoraTiles": None,
        "NumRedFives": [1, 0, 0],
        "IsTsumo": False,
        "WinTile": 0,
        "RoundWindTile": 27,
        "SelfWindTile": 27,
        "IsParent": False,
        "IsDaburii": False,
        "IsRiichi": False,
        "DiscardTiles": [],
        "LeftTiles34": [4, 4, 3, 3, 3, 3, 4, 3, 4, 
                        4, 4, 4, 4, 3, 3, 3, 3, 4, 
                        4, 3, 3, 4, 3, 3, 3, 4, 4, 
                        4, 4, 4, 4, 4, 4, 4],
        "LeftDrawTilesCount": 0,
        "NukiDoraNum": 0
    }
    player_info = result
    print(f"player_info: {player_info}")

    result = analysis_player_with_risk(player_info)
    print(f"Shanten: {result['shanten']}")
    print("Results14:")
    print(f"len results14: {len(result['results14'])}")
    for r in result['results14']:
        print(f"  Discard tile: {r['discardTile']}")
        print(f"  DiscardTileValue: {r['isDiscardDoraTile']}")
        print(f"  Shanten: {r['result13']['shanten']}")
        print(f"  Waits: {r['result13']['waits']}")
        print(f"  YakuTypes: {r['result13']['yakuTypes']}")
        print("  ---")
    print("IncShantenResults14:")
    print(f"len incShantenResults14: {len(result['incShantenResults14'])}")
    # for r in result['incShantenResults14']:
    #     print(f"  Discard tile: {r['discardTile']}")
    #     print(f"  Shanten: {r['result13']['shanten']}")
    #     print(f"  Waits: {r['result13']['waits']}")
    #     print("  ---")