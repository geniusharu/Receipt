import cv2
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
def allReceiptInfo(folder, rotate_folder):

    pict_names = os.listdir(folder) # 対象フォルダ内の画像ファイル名を取得
    res = {} # 最終結果の格納用

    for pict_name in tqdm(pict_names): #最初の5枚で確認するように変更してます
        try:
            pict_info = {} # 画像ごとの属性を格納用

            img_path = folder + pict_name # 画像のパスを生成
            receipt = ReceiptInfo(img_path, rotate_folder) # レシート属性のインスタンスを生成

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
#            pict_info['pict_name'] = pict_name

            res[pict_name] = pict_info
        except OSError:
            continue
        except cv2.error:
            continue

    return res

if __name__ == '__main__':
    output = 'submit.tsv'
    folder = "./test/"
    rotate_folder = './rotateimage_test/'
    valData = pd.read_csv('test.tsv', sep='\t') # 評価用データ

    alldata = allReceiptInfo(folder, rotate_folder)
    alldata = pd.DataFrame(alldata)
    alldata = alldata.T
    alldata.to_csv('test_data.csv')

    # submitfileの生成
    submit = []
    for f, p in zip(valData.file_name, valData.property):
        try:
            submit.append(alldata[p][f])
        except KeyError:
            # 欠損値の補間を追加しました。
            if p == 'cn_name':
                submit.append('ファミリマート')
            elif p == 'store_pref':
                submit.append('東京都')
            elif p == 'total_price':
                submit.append('663')
            else:
                submit.append('none')

    submit = pd.Series(submit).fillna('none')
    submit.to_csv(output, sep='\t', header=False, index=True)
