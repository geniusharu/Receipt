import numpy as np
import pyocr
import pyocr.builders

from datetime import datetime
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

        # 使用するツール（Tesseract）と言語（日本語）を指定
        self.tool = pyocr.get_available_tools()[0]
        self.lang = self.tool.get_available_languages()[1]

        # 画像データからテキストデータを抽出
        self.text = self.__getTextFromImage(self.img_path)

        # 都道府県のリスト　trainデータ内の個数順にソート済
        self.pref_list = ['東京都', '大阪府', '愛知県', '神奈川県', '千葉県', '埼玉県',
                          '兵庫県', '福岡県', '茨城県', '静岡県', '沖縄県', '三重県',
                          '京都府', '広島県', '鹿児島県', '岡山県', '宮城県', '長崎県',
                          '岐阜県', '北海道','栃木県', '奈良県', '長野県', '山口県',
                          '熊本県', '愛媛県', '滋賀県', '福島県', '石川県', '香川県',
                          '福井県', '岩手県', '和歌山県', '新潟県', '富山県', '大分県',
                          '群馬県', '宮崎県', '秋田県', '青森県','山形県', '佐賀県',
                          '山梨県', '徳島県', '高知県', '島根県', '鳥取県']

    # 画像データからテキストデータを取得
    def __getTextFromImage(self, img_path):
        text = self.tool.image_to_string(Image.open(img_path),
                                   lang=self.lang,
                                   builder=pyocr.builders.TextBuilder())
        return text

    # レシート種別
    def get_cn_name(self):
        """
        "ファミマ!!"、"ファミリマート"、"サークルK"、"サンクス"のいずれか
        """
        # TODO
        return "ファミリマート"

    # 店舗住所（都道府県）
    def get_store_pref(self):

        """
        "東京都"、"神奈川県"、等
        """

        store_pref = np.nan # 初期値はnanにしておく

        for pref in self.pref_list:
            if pref in self.text:
                store_pref = pref

        return store_pref

    # 店舗電話番号
    def get_store_tel(self):
        """
        ハイフン"-"なしの10桁の数字
        """
        # TODO
        return "1234567890"

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
        return 9999

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
        return "1234********5678"

    # 購入した商品の種類数
    def get_num_items(self):
        """
        整数値
        """
        # TODO
        return 1

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
    path = 'zswdmg51_000.jpg'
    receipt = ReceiptInfo(path)

    print('cn_name: ' + receipt.get_cn_name())
    print('store_pref: ' + receipt.get_store_pref())
    print('store_tel: ' + receipt.get_store_tel())
    print('bought_datetime: ' + receipt.get_bought_datetime())
    print('total_price: ' + str(receipt.get_total_price()))
    print('regi_number: ' + receipt.get_regi_number())
    print('duty_number: ' + receipt.get_duty_number())
    print('card_number: ' + receipt.get_card_number())
    print('num_items: ' + str(receipt.get_num_items()))
    print('items: ' + str(receipt.get_items()))