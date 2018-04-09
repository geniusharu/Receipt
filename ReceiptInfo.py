import cv2
import numpy as np
import os
import pandas as pd
import pyocr
import pyocr.builders
import re

from datetime import datetime
from keras.models import load_model
from PIL import Image

# 別途推定したCNNモデルをロード
CNN_MODEL_CN_NAME = load_model('./cn_name/CNN_cn_name.h5')
CNN_MODEL_NUM_ITEMS = load_model('./num_items/CNN_num_items.h5')

class ReceiptInfo(object):

    """
    レシートの属性情報を取得するためのクラス

    input:
        img_path: 元画像のパス（回転・トリミング前のもの）
        rotate_folder: 回転させた画像を保存しているフォルダ
    output:
        cn_name :レシート種別
        store_pref :店舗住所（都道府県）
        store_tel :店舗電話番号
        bought_datetime :商品購入年月日時
        total_price :合計金額
        regi_number :レジ番号
        duty_number :責任番号
        card_number :T-ポイントカード番号
        num_items :購入した商品の種類数
        items :購入した商品情報（キャンペーン対象のみ）

    """

    def __init__(self, img_path, rotate_folder):
        self.rotate_folder = rotate_folder
        self.img_path = self.__getRotateImage(img_path)

        # CNN用の画像サイズを指定
        self.image_size_w = 128
        self.image_size_h = 128

        # numpy array形式の画像データ（CNNモデル用）
        self.img_array_000 = self.__imagePreprocessing(self.img_path_000)
        self.img_array_090 = self.__imagePreprocessing(self.img_path_090)
        self.img_array_180 = self.__imagePreprocessing(self.img_path_180)
        self.img_array_270 = self.__imagePreprocessing(self.img_path_270)

        # 使用するツール（Tesseract）と言語（日本語）を指定
        self.tool = pyocr.get_available_tools()[0]
        self.lang = self.tool.get_available_languages()[1] # mita envでは[2]
        self.lang_en = self.tool.get_available_languages()[0]

        # 画像データからテキストデータを抽出
        self.text = self.__getTextFromImage(self.img_path)
        self.text_en = self.__getTextFromImageEn(self.img_path)

        # 都道府県のリスト　trainデータ内の個数順にソート済
        self.pref_list = ['東京都', '大阪府', '愛知県', '神奈川県', '千葉県', '埼玉県',
                          '兵庫県', '福岡県', '茨城県', '静岡県', '沖縄県', '三重県',
                          '京都府', '広島県', '鹿児島県', '岡山県', '宮城県', '長崎県',
                          '岐阜県', '北海道','栃木県', '奈良県', '長野県', '山口県',
                          '熊本県', '愛媛県', '滋賀県', '福島県', '石川県', '香川県',
                          '福井県', '岩手県', '和歌山県', '新潟県', '富山県', '大分県',
                          '群馬県', '宮崎県', '秋田県', '青森県','山形県', '佐賀県',
                          '山梨県', '徳島県', '高知県', '島根県', '鳥取県']

        # 店舗種別のリスト
        self.cn_name = ['ファミリマート', 'ファミマ!!', 'サンクス', 'サークルK']

        # 購入した商品の種類数のリスト
        self.num_items = [1,2,3,4]

        # キャンペーン対象商品のリスト
        self.item_list = pd.read_csv('item_list.csv')

    # 回転・トリミング後の画像パスを取得
    def __getRotateImage(self, img_path):
        __img = Image.open(img_path) #元画像を開く
        __img_array = np.asarray(__img) # numpy array形式に変換

        # 回転・トリミング後の画像パス
        self.img_path_000 = self.rotate_folder + img_path[-12:][:8] +"_000.jpg"
        self.img_path_090 = self.rotate_folder + img_path[-12:][:8] +"_090.jpg"
        self.img_path_180 = self.rotate_folder + img_path[-12:][:8] +"_180.jpg"
        self.img_path_270 = self.rotate_folder + img_path[-12:][:8] +"_270.jpg"

        # 縦幅<横幅なら270度回転させた画像、それ以外は回転させていない画像のパスを返す
        if __img_array.shape[0] < __img_array.shape[1]:
            return self.img_path_270
        else:
            return self.img_path_000

    # 画像データからテキストデータを取得
    def __getTextFromImage(self, img_path):
        text = self.tool.image_to_string(Image.open(img_path),
                                   lang=self.lang,
                                   builder=pyocr.builders.TextBuilder())
        return text

    def __getTextFromImageEn(self, img_path):
        text = self.tool.image_to_string(Image.open(img_path),
                                   lang=self.lang_en,
                                   builder=pyocr.builders.TextBuilder())
        return text

    # CNN用の画像前処理
    def __imagePreprocessing(self, filepath):
        img = cv2.imread(filepath)
        img = cv2.resize(img, (self.image_size_w, self.image_size_h)) #画像をリサイズ
        img = img.astype(float) / 255
        return img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    # テキストデータを整形の上、１行ごとに分けてリスト化して返します。
    def text_cleaner(self, text):
        text = text.replace("　", "")
        text = text.replace(" ", "")
        text = text.replace("（", "(")
        text = text.replace("）", ")")
        result = [ line for line in text.split('\n') if len(line)>0 ]
        return result

    # text_cleanerのアウトプット（リスト）から、ファミマ文字列があるか確認し、あればTrueを、なければFalseを返します。
    def check_picture(self, list):
        moji = ' '.join(list[:5]) # listの最初の5要素を結合して文字列を生成
        res =  "Fam" in moji or "ami" in moji or "art" in moji
        return res

    def gross_amount_check(self, list):
        cnt = 0
        # 合計と記載された行が何行目かを cnt で調べる
        for line in list:
            if line.find('合') > -1 and line.find('算') and line.find('商') == -1 and line.find('品') == -1 and line.find('値') == -1 and line.find('引') == -1 :
                break
            cnt = cnt + 1
        try:
            tgt_row = list[cnt]
            tgt_row = tgt_row.replace('フ','7')
            tgt_row = tgt_row.replace('了','7')
            my_result = ''.join([_l if _l.isdigit() else '' for _l in tgt_row])
        except: # わからないときは663円（最頻値）と予想
            my_result = 663
        return my_result

    def regi_number(self, list):
        cnt = 0
        my_result = ''
        for line in list:
            if line.find('レシ')>-1 or line.find('レジ')>-1:
                regi = [l if l.isdigit() else '' for l in line]
                tmp = ''.join(regi)
                if len(tmp) >= 5:
                    my_result = '{0}{1}{2}'.format(tmp[:1], '-', tmp[1:5])
        if my_result:
            return my_result
        else:
            for line in list:
                if line.find('N')>-1:
                    break
                cnt = cnt + 1
            regi = [l if l.isdigit() else '' for l in  list[cnt] ]
            tmp = ''.join(regi)
            if len(tmp) >= 5:
                my_result = '{0}{1}{2}'.format(tmp[:1], '-', tmp[1:5])

            if my_result:
                return my_result
            else:
                'none'

    # レシート種別
    def get_cn_name(self):

        """
        "ファミマ!!"、"ファミリマート"、"サークルK"、"サンクス"のいずれか
        """

        # 各画像ごとに各クラスに分類される確率を算出
        predict_000 = CNN_MODEL_CN_NAME.predict_proba(self.img_array_000)
        predict_090 = CNN_MODEL_CN_NAME.predict_proba(self.img_array_090)
        predict_180 = CNN_MODEL_CN_NAME.predict_proba(self.img_array_180)
        predict_270 = CNN_MODEL_CN_NAME.predict_proba(self.img_array_270)
        probas = [predict_000, predict_090, predict_180, predict_270]

        # 各画像の平均確率を算出
        average_proba = np.average(probas, axis=0)

        # 確率が最大となるクラスを選択
        predict_class = average_proba.argmax(axis=-1)[0]

        return self.cn_name[predict_class]

    # 店舗住所（都道府県）
    def get_store_pref(self):

        """
        "東京都"、"神奈川県"、等
        """

        store_pref = '' # 初期値は空白にしておく

        # テキスト内に都道府県名の文字列があればそれを返す
        for pref in self.pref_list:
            if pref in self.text:
                store_pref = pref
                break

        # nanの場合は"東京都"を返す
        if store_pref:
            return store_pref
        else:
            return '東京都'

    # 店舗電話番号
    def get_store_tel(self):

        """
        ハイフン"-"なしの10桁の数字
        """

        txt = self.text.split('\n') # '話'で抜けそうな感じだったので日本語使用に変更しています。

        store_tel = '' # 初期値は空白に

        for l in txt:
            # "話"だけ抜けてる場合が多いのでこれで判別します。
            if '話' in l:
                _store_tel = [_l if _l.isdigit() else '' for _l in l] # リスト内の整数値のみ抽出
                _store_tel = ''.join(_store_tel)
                if len(_store_tel) >=10:
                    store_tel = _store_tel[:11]

        if store_tel:
            return store_tel
        else:
            return 'none'

    # 商品購入年月日時
    def get_bought_datetime(self):

        """
        yyyy-mm-dd hh:mm:ss
        """

        d = '' #初期値は空白にしておく

        # 整形後のテキストデータを取得
        txt = self.text.split('\n')

        # テキスト内で"年月日:"が含まれる行を探索
        for t in txt:
            try:
                # ファミリーマートタイプを想定
                if '年' in t and '月' in t and '日' in t and ':' in t:
                        _t = re.split('[年月日:]', t) # 年月日:でsplitしたリストを生成

                        year = int(_t[0][-4:].strip())
                        month = int(_t[1][-2:].strip())
                        day = int(_t[2][-2:].strip())
                        hour = int(_t[-2][-2:].strip())
                        minute = int(_t[-1][:2].strip())

                        if len(str(year)) != 4:
                            year = 2017 #うまく読み込めてない場合は2017に変更

                        d = datetime(year, month, day, hour=hour, minute=minute)
                        break
                # サークルＫタイプを想定
                elif '年' in t and '月' in t and '日' in t and '時' in t and '分' in t:
                        _t = re.split('[年月日時分]', t) # 年月日時分でsplitしたリストを生成

                        year = int(_t[0][-4:].strip())
                        month = int(_t[1][-2:].strip())
                        day = int(_t[2][-2:].strip())
                        hour = int(_t[-3][-2:].strip())
                        minute = int(_t[-2][:2].strip())

                        if len(str(year)) != 4:
                            year = 2017 #うまく読み込めてない場合は2017に変更

                        d = datetime(year, month, day, hour=hour, minute=minute)
                        break
            except ValueError:
                continue
            except IndexError:
                continue

        # dがNoneでなければそのまま返す
        if d:
            bought_datetime = d
        else:
            # TODO
            # 年月日の文字列が読み込めない場合に数値から日付を推定するロジックがあると良いかもです
            bought_datetime = datetime(2017, 10, 30, hour=12, minute=10) #これが最頻値っぽいです

        return bought_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # 合計金額
    def get_total_price(self):
        """
        整数値
        """
        text_jp = self.text
        text_jp = self.text_cleaner(text_jp)
        gross_amount = self.gross_amount_check(text_jp)
        return gross_amount

    # レジ番号
    def get_regi_number(self):
        """
        i-iiii　iには0-9の数字が入る。画像に存在しない場合は"none"とする。
        """
        text = self.text
        text = self.text_cleaner(text)
        try:
            regi_number = self.regi_number(text)
        except:
            regi_number = 'none'
        return regi_number

    # 責任番号
    def get_duty_number(self):

        """
        iii　iには0-9の数字が入る。画像に存在しない場合は"none"とする。
        """

        duty_number = '' #初期値は空白にしておく

        # 整形後のテキストデータを取得
        txt = self.text.split('\n')

        # テキスト内で"No"が含まれる行を探索
        for t in txt:
            if 'No' in t:
                try:
                    duty_number = t.split('No')[1]
                    duty_number = ''.join([_l if _l.isdigit() else '' for _l in duty_number]) #数値のみを抽出
                    break
                except ValueError:
                    continue
                except IndexError:
                    continue

        if duty_number and len(duty_number)==3: #3桁でない場合はうまく抜けてない可能性が高いのでnoneで返すように変更
            return duty_number
        else:
            return 'none'

    # T-ポイントカード番号
    def get_card_number(self):

        """
        iiii********iiii　iには0-9の数字が入る。画像に存在しない場合は"none"とする。
        """

        text_jp = self.text
        text_jp = self.text_cleaner(text_jp)
        text_en = self.text_en
        text_en = self.text_cleaner(text_en)

        card_number = '' # 初期値は空白にしておく

        for l in text_jp:
            # "********"を完璧に抜けているパターンは少ないので数を減らしています。
            if '**' in l or '対象' in l or '会員' in l or '番号' in l:
                _card_number = [_l if _l.isdigit() else '' for _l in l] # リスト内の整数値のみ抽出
                _card_number = ''.join(_card_number)
                if len(_card_number) >= 8:
                    card_number = str(_card_number[:4]) + '********' + str(_card_number[-4:])

        # card_numberが入ってればそれを返して、入っていなければ英語版を読みに行く
        if card_number:
            return card_number
        else:
            for l in text_en:
                if '**' in l: # "********"を完璧に抜けているパターンは少ないので数を減らしています。
                    _card_number = [_l if _l.isdigit() else '' for _l in l] # リスト内の整数値のみ抽出
                    _card_number = ''.join(_card_number)
                    if len(_card_number) >= 8:
                        card_number = str(_card_number[:4]) + '********' + str(_card_number[-4:])
            if card_number:
                return card_number
            else:
                return 'none'

        return card_number

    # 購入した商品の種類数
    def get_num_items(self):

        """
        整数値
        """

        # 各画像ごとに各クラスに分類される確率を算出
        predict_000 = CNN_MODEL_NUM_ITEMS.predict_proba(self.img_array_000)
        predict_090 = CNN_MODEL_NUM_ITEMS.predict_proba(self.img_array_090)
        predict_180 = CNN_MODEL_NUM_ITEMS.predict_proba(self.img_array_180)
        predict_270 = CNN_MODEL_NUM_ITEMS.predict_proba(self.img_array_270)
        probas = [predict_000, predict_090, predict_180, predict_270]

        # 各画像の平均確率を算出
        average_proba = np.average(probas, axis=0)

        # 確率が最大となるクラスを選択
        predict_class = average_proba.argmax(axis=-1)[0]

        return self.num_items[predict_class]

    # 購入した商品情報（キャンペーン対象のみ）
    def get_items(self):

        """
        [(商品名_1,単価_1,値引き_1,個数_1),...,(商品名_n,単価_n,値引き_n,個数_n)]
        ※存在しない場合は"none"とする。単価と個数は整数値、値引き値は-を付け、ない場合は0とする。
        """

        items = []

        # テキスト内にitem listと一致する文字列があれば商品情報を追加。
        for item_name in self.item_list['name']:
            if item_name in self.text:
                __price = self.item_list[self.item_list.name==item_name]['price'].iloc[0] #当該商品の価格を抽出
                __discount = 0 # とりあえず全て値引きは0円に固定
                __num_items = 1 # とりあえず個数は1個に固定
                item = (item_name, __price, __discount, __num_items)
                items.append(item)

        # itemsが空の場合はnoneを返す
        if items:
            return items
        else:
            return 'none'

if __name__ == '__main__':
    # test
    img_path = './test/a06jut8e.jpg'
    rotate_folder = './rotateimage_test/'

    receipt = ReceiptInfo(img_path, rotate_folder)

    print('cn_name: ' + receipt.get_cn_name())
    print('store_pref: ' + str(receipt.get_store_pref()))
    print('store_tel: ' + str(receipt.get_store_tel()))
    print('bought_datetime: ' + receipt.get_bought_datetime())
    print('total_price: ' + str(receipt.get_total_price()))
    print('regi_number: ' + receipt.get_regi_number())
    print('duty_number: ' + receipt.get_duty_number())
    print('card_number: ' + str(receipt.get_card_number()))
    print('num_items: ' + str(receipt.get_num_items()))
    print('items: ' + str(receipt.get_items()))

"""
    pict_name_list = os.listdir('./train2')
    for pict_name in sorted(pict_name_list):
        if pict_name.find(".jpg") > -1 and pict_name.find('_270') > -1:
            path = './rotateimage/' + pict_name
            receipt = ReceiptInfo(path)
            print('cn_name: ' + receipt.get_cn_name())
            print('store_pref: ' + str(receipt.get_store_pref()))
            print('store_tel: ' + str(receipt.get_store_tel()))
            print('bought_datetime: ' + receipt.get_bought_datetime())
            print('total_price: ' + str(receipt.get_total_price()))
            print('regi_number: ' + receipt.get_regi_number())
            print('duty_number: ' + receipt.get_duty_number())
            print('card_number: ' + str(receipt.get_card_number()))
            print('num_items: ' + str(receipt.get_num_items()))
            print('items: ' + str(receipt.get_items()))
        else:
            pass
"""
