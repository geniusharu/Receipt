import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from ReceiptInfo import ReceiptInfo

"""
画像データからレシートの属性情報を抽出し、集計するためのスクリプト。
別途実装したReceiptInfoモジュールを使用。
"""

# レシート情報の集計用
def aggregateReceiptInfo(folder):

    pict_names = os.listdir(folder) # 対象フォルダ内の画像ファイル名を取得
    res = {} # 最終結果の格納用

    for pict_name in tqdm(pict_names): #最初の5枚で確認するように変更してます
        pict_info = {} # 画像ごとの属性を格納用

        img_path = folder + pict_name # 画像のパスを生成
        receipt = ReceiptInfo(img_path) # レシート属性のインスタンスを生成

        # 各項目の推定結果を保存
        pict_info['cn_name'] = receipt.get_cn_name()
        pict_info['store_pref'] = receipt.get_store_pref()
        pict_info['store_tel'] = receipt.get_store_tel()
        pict_info['bought_datetime'] = receipt.get_bought_datetime()
        pict_info['total_price'] = receipt.get_total_price()
        pict_info['regi_number'] = receipt.get_regi_number()
        pict_info['duty_number'] = receipt.get_duty_number()
        pict_info['card_number'] = receipt.get_card_number()
        pict_info['num_items'] = receipt.get_num_items()
        pict_info['items'] = receipt.get_items()
        pict_info['pict_name'] = pict_name
        pict_info['original_filename'] = pict_name[:-8] + ".jpg"
        pict_info['text'] = receipt.text

        res[pict_name] = pict_info

#    res = pd.DataFrame(res)
#    res = res.T
#    res.to_csv(output)

    return res

# 提出用のtsvファイル生成用
def generateSubmitFile(output,folder,valDatapath):

    valData = pd.read_csv(valDatapath, sep='\t') # 評価用データの読み込み
    allReceiptInfo = aggregateReceiptInfo(folder) # testフォルダ以下のレシート情報の集計

    submit = []
    for f, p in tqdm(zip(valData.file_name, valData.property)):
        submit.append(allReceiptInfo[f][p])

    submit = pd.Series(submit)
    submit.to_csv(output, sep='\t', header=False, index=True)

    return submit

if __name__ == '__main__':
#    folder = "./rotateimage/"
#    aggregateReceiptInfo(folder, output)
    output = 'submit.tsv'
    folder = "./test/"
    valDatapath = 'test.tsv'
    generateSubmitFile(output,folder,valDatapath)
