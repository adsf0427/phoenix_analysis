import json
import requests
import gzip

# mjai = '''{"type":"start_kyoku","bakaze":"E","dora_marker":"5mr","kyoku":2,"honba":1,"kyotaku":0,"oya":1,"scores":[21400,44800,10100,23700],"tehais":[["E","4m","5sr","2m","1s","6s","4m","3s","N","1s","C","W","1s"],["E","N","2s","9p","9p","6p","4p","W","8s","6p","4m","C","7p"],["3m","8p","5s","3m","5m","3s","7m","3p","P","3p","2s","7m","2p"],["8s","9s","5m","2m","6m","8p","E","8m","4s","1m","P","4p","C"]]}
# {"type":"tsumo","actor":1,"pai":"9p"}
# {"type":"dahai","actor":1,"pai":"8s","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"8s"}
# {"type":"dahai","actor":2,"pai":"P","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"9p"}
# {"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"1p"}
# {"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
# {"type":"tsumo","actor":1,"pai":"E"}
# {"type":"dahai","actor":1,"pai":"4m","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"5p"}
# {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"3m"}
# {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"1m"}
# {"type":"dahai","actor":0,"pai":"C","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"8p"}
# {"type":"dahai","actor":1,"pai":"C","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"8p"}
# {"type":"dahai","actor":2,"pai":"5s","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"8s"}
# {"type":"dahai","actor":3,"pai":"E","tsumogiri":false}
# {"type":"pon","actor":1,"target":3,"pai":"E","consumed":["E","E"]}
# {"type":"dahai","actor":1,"pai":"2s","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"3s"}
# {"type":"dahai","actor":2,"pai":"5p","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"7s"}
# {"type":"dahai","actor":3,"pai":"8s","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"7p"}
# {"type":"dahai","actor":0,"pai":"E","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"C"}
# {"type":"dahai","actor":1,"pai":"C","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"7m"}
# {"type":"dahai","actor":2,"pai":"5m","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"2p"}
# {"type":"dahai","actor":3,"pai":"8m","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"P"}
# {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
# {"type":"tsumo","actor":1,"pai":"9m"}
# {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"6s"}
# {"type":"dahai","actor":2,"pai":"6s","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"9m"}
# {"type":"dahai","actor":3,"pai":"9m","tsumogiri":true}
# {"type":"tsumo","actor":0,"pai":"S"}
# {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"5s"}
# {"type":"dahai","actor":1,"pai":"5s","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"S"}
# {"type":"dahai","actor":2,"pai":"S","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"2m"}
# {"type":"dahai","actor":3,"pai":"2m","tsumogiri":true}
# {"type":"tsumo","actor":0,"pai":"9m"}
# {"type":"dahai","actor":0,"pai":"9m","tsumogiri":true}
# {"type":"tsumo","actor":1,"pai":"7s"}
# {"type":"dahai","actor":1,"pai":"7s","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"8m"}
# {"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"2m"}
# {"type":"dahai","actor":3,"pai":"2m","tsumogiri":true}
# {"type":"tsumo","actor":0,"pai":"W"}
# {"type":"dahai","actor":0,"pai":"S","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"3p"}
# {"type":"dahai","actor":1,"pai":"W","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"1p"}
# {"type":"dahai","actor":2,"pai":"2s","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"5pr"}
# {"type":"dahai","actor":3,"pai":"2p","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"4p"}
# {"type":"dahai","actor":0,"pai":"N","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"3s"}
# {"type":"dahai","actor":1,"pai":"3s","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"7s"}
# {"type":"dahai","actor":2,"pai":"7s","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"9m"}
# {"type":"dahai","actor":3,"pai":"4s","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"6s"}
# {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"5m"}
# {"type":"dahai","actor":1,"pai":"5m","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"6s"}
# {"type":"dahai","actor":2,"pai":"6s","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"4p"}
# {"type":"dahai","actor":3,"pai":"9m","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"1s"}
# {"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"F"}
# {"type":"dahai","actor":1,"pai":"N","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"4s"}
# {"type":"dahai","actor":2,"pai":"4s","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"4m"}
# {"type":"dahai","actor":3,"pai":"8p","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"5p"}
# {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"P"}
# {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"1p"}
# {"type":"dahai","actor":2,"pai":"7m","tsumogiri":false}
# {"type":"tsumo","actor":3,"pai":"1p"}
# {"type":"dahai","actor":3,"pai":"1p","tsumogiri":true}
# {"type":"tsumo","actor":0,"pai":"6p"}
# {"type":"dahai","actor":0,"pai":"6s","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"5s"}
# {"type":"dahai","actor":1,"pai":"5s","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"8m"}
# {"type":"reach","actor":2}
# {"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
# {"type":"reach_accepted","actor":2}
# {"type":"tsumo","actor":3,"pai":"3p"}
# {"type":"dahai","actor":3,"pai":"9s","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"5p"}
# {"type":"ankan","actor":0,"consumed":["1s","1s","1s","1s"]}
# {"type":"dora","dora_marker":"7m"}
# {"type":"tsumo","actor":0,"pai":"2p"}
# {"type":"dahai","actor":0,"pai":"3s","tsumogiri":false}
# {"type":"tsumo","actor":1,"pai":"9s"}
# {"type":"dahai","actor":1,"pai":"9s","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"W"}
# {"type":"dahai","actor":2,"pai":"W","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"2p"}
# {"type":"dahai","actor":3,"pai":"8s","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"S"}
# {"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
# {"type":"tsumo","actor":1,"pai":"7p"}
# {"type":"dahai","actor":1,"pai":"8p","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"2s"}
# {"type":"dahai","actor":2,"pai":"2s","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"4s"}
# {"type":"dahai","actor":3,"pai":"7s","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"N"}
# {"type":"dahai","actor":0,"pai":"N","tsumogiri":true}
# {"type":"tsumo","actor":1,"pai":"F"}
# {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
# {"type":"tsumo","actor":2,"pai":"F"}
# {"type":"dahai","actor":2,"pai":"F","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"F"}
# {"type":"dahai","actor":3,"pai":"4s","tsumogiri":false}
# {"type":"chi","actor":0,"target":3,"pai":"4s","consumed":["5sr","6s"]}
# {"type":"dahai","actor":0,"pai":"5p","tsumogiri":false}
# {"type":"chi","actor":1,"target":0,"pai":"5p","consumed":["3p","4p"]}
# {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
# {"type":"tsumo","actor":2,"pai":"7s"}
# {"type":"dahai","actor":2,"pai":"7s","tsumogiri":true}
# {"type":"tsumo","actor":3,"pai":"6m"}
# {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
# {"type":"tsumo","actor":0,"pai":"8m"}
# {"type":"dahai","actor":0,"pai":"8m","tsumogiri":true}
# {"type":"tsumo","actor":1,"pai":"9s"}
# {"type":"dahai","actor":1,"pai":"9s","tsumogiri":true}
# {"type":"ryukyoku","deltas":[1000,1000,1000,-3000]}
# {"type":"end_kyoku"}'''

if __name__ == '__main__':
    with open('input/mjai.json', 'r') as f:
        mjai = f.read()
    post_data = {'data': mjai.split('\n')}
    data = json.dumps(post_data, separators=(',', ':'))
    compressed_data = gzip.compress(data.encode('utf-8'))
    player_id = 1

    # @app.route("/engine_4p/<string:engine_name>/<string:player_id>", methods=['GET'])
    result = requests.get('http://122.51.149.46:9875/engine_4p' + '/0/' + str(player_id), data=compressed_data)

    result_log = json.loads(gzip.decompress(result.content))['data']
    # result = json.loads(result_log)

    print(result_log)
    print(len(result_log))
    # print(result_log[0])