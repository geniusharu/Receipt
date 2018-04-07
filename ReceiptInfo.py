import cv2
import numpy as np
import os
import pyocr
import pyocr.builders

from datetime import datetime
from keras.models import load_model
from PIL import Image

class ReceiptInfo(object):

    """
    レシートの属性情報を取得するためのクラス

    input:
        img_path: 画像のパス

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

    def __init__(self, img_path):
        self.img_path = img_path

        # CNN用の画像サイズを指定
        self.image_size_w = 128
        self.image_size_h = 128

        # numpy array形式の画像データ
        self.img_array = self.__imagePreprocessing(self.img_path)

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
    # 多分これで同じ結果になると思います。
    def check_picture(self, list):
        moji = ' '.join(list[:5]) # listの最初の5要素を結合して文字列を生成
        res =  "Fam" in moji or "ami" in moji or "art" in moji
        return res

    # Kuso-codeをfixして高速化
    def phone_number_check(self, list):しょうひんすうｒ
        for l in list[:10]:
            phone_number = [_l if _l.isdigit() else '' for _l in l] # リスト内の整数値のみ抽出
            phone_number = ''.join(phone_number)
            if len(phone_number) >=10:
                return phone_number[:11]

    def card_number_check(self, list_jp, list_en):
        for l in list_jp:
            if l.find('********') > -1:
                phone_number = [_l if _l.isdigit() else '' for _l in l] # リスト内の整数値のみ抽出
                phone_number = ''.join(phone_number)
                if len(phone_number) >=8:
                    my_result = l[-16:]
        # my_resultが入ってればそれを返して、入っていなければ英語版を読みに行く
        try:
            my_result
            return my_result
        except:
            for l in list_en:
                if l.find('********') > -1:
                    phone_number = [_l if _l.isdigit() else '' for _l in l] # リスト内の整数値のみ抽出
                    phone_number = ''.join(phone_number)
                    if len(phone_number) >=8:
                        return l[-16:]

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
        except: # わからないときは108円と予想
            my_result = 663
        return my_result

    # レシート種別
    def get_cn_name(self):

        """
        "ファミマ!!"、"ファミリマート"、"サークルK"、"サンクス"のいずれか
        """

        # 別途推定したモデルをロード
        model = load_model('./cn_name/CNN_cn_name.h5')

        # モデルの推定値を求める
        predict = model.predict_classes(self.img_array)[0]

        return self.cn_name[predict]

    # 店舗住所（都道府県）
    def get_store_pref(self):

        """
        "東京都"、"神奈川県"、等
        """

        store_pref = np.nan # 初期値はnanにしておく

        # テキスト内に都道府県名の文字列があればそれを返す
        for pref in self.pref_list:
            if pref in self.text:
                store_pref = pref
                break

        return store_pref

    # 店舗電話番号
    def get_store_tel(self):
        """
        ハイフン"-"なしの10桁の数字
        """
        text = self.text_en  # language は英語を使用
        text = self.text_cleaner(text)
        store_tel = self.phone_number_check(text)
        return store_tel

    # 商品購入年月日時
    def get_bought_datetime(self):
        """
        yyyy-mm-dd hh:mm:ss
        """
        # TODO
        d = datetime.now() # とりあえず現在時刻を返すようにしてます
        return d.strftime("%Y-%m-%d %H:%M:%S")

    # 合計金額
    def get_total_price(self):
        """
        整数値
        """
        # TODO
        text_jp = self.text
        text_jp = self.text_cleaner(text_jp)
        gross_amount = self.gross_amount_check(text_jp)
        return gross_amount

    # レジ番号
    def get_regi_number(self):
        """
        i-iiii　iには0-9の数字が入る。画像に存在しない場合は"none"とする。
        """
        #TODO
        return "1-2345"

    # 責任番号
    def get_duty_number(self):
        """
        iii　iには0-9の数字が入る。画像に存在しない場合は"none"とする。
        """
        # TODO
        return "123"

    # T-ポイントカード番号
    def get_card_number(self):
        """
        iiii********iiii　iには0-9の数字が入る。画像に存在しない場合は"none"とする。
        """
        # TODO
        text_jp = self.text
        text_jp = self.text_cleaner(text_jp)
        text_en = self.text_en
        text_en = self.text_cleaner(text_en)
        card_number = self.card_number_check(text_jp,text_en)
        return card_number

    # 購入した商品の種類数
    def get_num_items(self):

        """
        整数値
        """

        # 別途推定したモデルをロード
        model = load_model('./num_items/CNN_num_items.h5')

        # モデルの推定値を求める
        predict = model.predict_classes(self.img_array)[0]

        return self.num_items[predict]

    # 購入した商品情報（キャンペーン対象のみ）
    def get_items(self):
        """
        [(商品名_1,単価_1,値引き_1,個数_1),...,(商品名_n,単価_n,値引き_n,個数_n)]
        ※存在しない場合は"none"とする。単価と個数は整数値、値引き値は-を付け、ない場合は0とする。
        """
        # TODO
        return [('手巻シーチキンマヨネー',110,0,1)]

if __name__ == '__main__':
    # test
#    path = 'rotateimage/a01e2wt2_270.jpg'
#    path = 'zswdmg51_000.jpg' これでテストしたけど日本語のほうがうまく電話番号抜けました
#    receipt = ReceiptInfo(path)

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
