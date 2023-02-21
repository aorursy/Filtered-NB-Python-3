#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('time', '', '#  このセル（jupyter でのプログラムの実行単位です。）は、ライブラリを読み込み、sqlite データと接続して、一時的なテーブルを作成します。\n# pandas というデータを扱うライブラリを読み込みます。\n# このプログラムでは、データ操作はＳＱＬを通じて行うので、pandas は、グラフを書くために使うのが主です。\nimport pandas as pd # sql の検索結果を保存する\nimport numpy as np  # pandas に付随するもの、1次元の操作の時に使う\nimport pandas.io.sql as psql # sql と pandas を結びつける\n\n# sqliteのライブラリ\nimport sqlite3\n\n# 線形回帰\u3000使う頻度はあまりないですが、とりあえず、いれておきます。\nfrom sklearn import linear_model\nclf = linear_model.LinearRegression()\n\n# HTMLで表示する\u3000エクセルにコピペするときに便利かも\nfrom IPython.display import display, HTML\n\n# markdown 用 qiita に\nfrom tabulate import tabulate\n\n# 日時を扱う\nfrom datetime import datetime as dt\nimport time\n\n# グラフ\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nfrom matplotlib import ticker\n%matplotlib inline\n\n# system 関係のライブラリ\nimport sys\n# os の機能を使うライブラリ\nimport os\n\n# 正規表現\nimport re\n\n# json,yaml 形式を扱う\nimport json\nimport yaml\n\n# 変数の状態を調べる\nimport inspect\n\n# 文字コード\u3000日本語の文字コードが sjis のときに役立ちます。\nimport codecs\n\n# Web からデータを取得する\nimport requests\n\n# 貿易統計のデータ\n# http://www.customs.go.jp/toukei/info/tsdl_e.htm\n# コード\u3000輸出は日本語のみ\n# https://www.customs.go.jp/toukei/sankou/code/code_e.htm \n\n# sqlite に show tables がないので補足するもの\nshow_tables = "select tbl_name from sqlite_master where type = \'table\'"\n# describe もないで、補完します。\ndesc = "PRAGMA table_info([{table}])"\n# メモリで、sqlite を使います。kaggle のスクリプト上では、オンメモリでないと新規テーブルがつくれません\n# プログラムの一行が長いときは\u3000\\\u3000で改行します。\nconn = \\\n    sqlite3.connect(\':memory:\')\n\n# sql を実行するための変数です。\ncursor = conn.cursor()\n\n# 1997 年から、2019 年までの年ベースのデータです。テーブルは、year_from_1997 \n# year_from_1997\nattach = \'attach "../input/japan-trade-statistics/y_1997.db" as y_1997\'\ncursor.execute(attach)\n\n# 2018 年の月別集計 テーブル名も ym_2018 \nattach = \'attach "../input/japan-trade-statistics/ym_2018.db" as ym_2018\'\ncursor.execute(attach)\n\n# 2019 年の月別集計 テーブル名も ym_2019\nattach = \'attach "../input/japan-trade-statistics/ym_2019.db" as ym_2019\'\ncursor.execute(attach)\n\n# 2020 年の月別集計 テーブル名も ym_2020\nattach = \'attach "../input/japan-trade-statistics/ym_2020.db" as ym_2020\'\ncursor.execute(attach)\n\n# hs code,country,HSコードです。使いやすいように pandas\u3000に変更しておきます。\nattach = \'attach "../input/japan-trade-statistics/codes.db" as code\'\ncursor.execute(attach)\n# import hs,country code as pandas\ntmpl = "{hs}_{lang}_df =  pd.read_sql(\'select * from code.{hs}_{lang}\',conn)"\nfor hs in [\'hs2\',\'hs4\',\'hs6\',\'hs6\',\'hs9\']:\n    for lang in [\'jpn\',\'eng\']:\n        exec(tmpl.format(hs=hs,lang=lang))        \n\n# 国コードも pandas で扱えるようにします。\n# country table: country_eng,country_jpn\ncountry_eng_df = pd.read_sql(\'select * from code.country_eng\',conn)\ncountry_eng_df[\'Country\']=country_eng_df[\'Country\'].apply(str)\ncountry_jpn_df = pd.read_sql(\'select * from code.country_jpn\',conn)\ncountry_jpn_df[\'Country\']=country_jpn_df[\'Country\'].apply(str)\n\n# custom  table: code.custom 税関別のコードです\ncustom_df = pd.read_sql(\'select * from code.custom\',conn)\nattach = \'attach "../input/japan-trade-statistics/custom_from_2012.db" as custom_from\'\ncursor.execute(attach)\nattach = \'attach "../input/custom-2016/custom_2018.db" as custom_2018\'\ncursor.execute(attach)\nattach = \'attach "../input/custom-2016/custom_2019.db" as custom_2019\'\ncursor.execute(attach)\n\nattach = \'attach "../input/japan-trade-statistics/custom_2020.db" as custom_2020\'\ncursor.execute(attach)\n\n# 計算時間を節約するために、年のデータから、2019 年を切り出します。\n# 最初のはエラー処理です。y_2019 というテーブルが存在すると、新規に y_2019 を作ろうとするとエラーになります。\n# error の場合は、何もせず、次にすすみます。\ntry:\n    cursor.execute(\'drop table y_2019\')\nexcept:\n    pass\n\n# これからが、SQl になります。複数行で書くことが多いのでsql という変数に複数行を代入します。\n# 最後の [1:-1] は、一行目（改行で空白）と最後の行（これも改行だけで空白）をとりのぞくためです。\n# 0 から始まるので、1 だと、２行目から最後の行のひとつ手前までです。\nsql = """\ncreate table y_2019 \nas select * from year_from_1997\nwhere Year = 2019\n"""[1:-1]\n# 上記の sql を実行しして、2019 年のデータをつくります。\ncursor.execute(sql)\nconn.commit()\n# sql の説明です。\n# create table テーブル名 : テーブルを新規作成\u3000ここでは、y_2019 \n# as select * from  テーブル名\u3000: テーブル名(year_from_1997)からつくります。\n# where Year = 2019 : 2019 年のデータを指定します。Year は、数値なので、2019 と書きます。')




# 便利なクラス（sql 実行 + グラフ）ut.関数名で使います。
class util():
    def sql(self,sql):
        # sql で抽出されたデータを pandas の形式で戻します。
        return(pd.read_sql(sql,conn))
 
    # 折れ線グラフ 一系列　色は、b ( blue )
    def g1(self,df,x,y,color='b'):
        plt.figure(figsize=(20, 10))
        ax = sns.lineplot(x=x,y=y,data=df,linewidth=7.0,color=color)
        # これは、x軸（時系列）の単位が省略されないようにする設定
        # 何もしないと、2000,2005,2010のように一年分がとばされてしまいます。
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) 
        # 以下は、グラフ表示と表示終了
        plt.show()
        plt.close()
        
    # 折れ線グラフ 二系列　主に輸出入　比較につかいます。
    # hue は項目（輸出、輸入）
    # 輸出がb ( blue )、輸入がr ( red )
    # 指定例 ut.g2(df,'ym','Value','exp_imp')
    def g2(self,df,x,y,hue,palette={1: "b", 2: "r"}):
        plt.figure(figsize=(20, 10))
        ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,
             palette=palette,
             data=df)
        # 凡例の位置　２は左上
        ax.legend_._loc = 2
        # 目盛り（年や、年月を省略しない設定）
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
        plt.close()
        
    # 複数系列の折れ線グラフ 
    def gx(self,df,x,y,hue,palette={}):
        plt.figure(figsize=(20, 10))
        if palette == {}:
            ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,data=df)
        else:
            ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,palette=palette,data=df)
        # 凡例の位置　２は左上
        ax.legend_._loc = 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))    
        
    def bar(self,df,y,x,prefix='',color='b'):
        # 色見本
        #https://matplotlib.org/examples/color/named_colors.html
        # 色の意味　合計: gold     輸出: b (blue) 輸入: r ( red )　をつかっています。
        if len(prefix) > 0:
            df[y] = df[y].map(lambda x: 'hs' + str(x))
        ax = sns.barplot(y=y, x=x, data=df,color=color)
        plt.show()
        plt.close()


    # 輸出入コードのurl を表示する
    # 2種類だします。
    def hs_url(self,hs_code,exp_imp=2,yyyy_mm='2020_4'):
        # 最新貿易統計
        # 輸出 https://www.customs.go.jp/yusyutu/index.htm
        # 輸入 https://www.customs.go.jp/tariff/index.htm
        hs = hs_code[0:2]
      
        if exp_imp == 1:
            ex = 'yusyutu'
        else:
            ex = 'tariff'
        tmpl = 'https://www.customs.go.jp/{ex}/{yyyy_mm}/data/print_j_{hs}.htm'
        print(tmpl.format(ex=ex,yyyy_mm=yyyy_mm,hs=hs))
        
        # HS 4桁早見表（最新でないことがあります。）
        # 
        tmpl = 'https://www.toishi.info/hscode/{ths}/{ths4}.html'

        ths = str(int(hs))
        ths4 = hs_code[0:4]
        print(tmpl.format(ths=ths,ths4=ths4))
                

    # db との接続 conn は、global 変数として使う 
    def hs_table_create(self,hs_code,tables=['y_2019','year_from_1997','ym_2018_2020']):
        
        if len(hs_code) not in (2,4,6,9):
            print(hs_code + ': 桁数がおかしいです。')
            return
        
        hs = 'hs' + str(len(hs_code))
        
        sql = """
        create table hs{hs_code}_{table}
        as select * from {table}
        where {hs} = '{hs_code}'
        """[1:-1]
        
        for table in tables:
            tg = 'drop table hs{hs_code}_{table}'.format(hs_code=hs_code,table=table)
            print(tg)
            try:
                cursor.execute(tg)
            except:
                pass
            cursor.execute(sql.format(hs=hs,hs_code=hs_code,table=table))

        conn.commit()
        
    def hs_name_get(self,hs_code):
        hs = len(hs_code)
        if hs not in (2,4,6,9):
            print('HS コードの長さがまちがっています。 ' + str(hs))
        hs = str(hs)
        print(hs_code)
        text = 'hs' + hs + '_eng_df.query(' +"'"+ 'hs' + hs + '=="' + hs_code + '"' + "')"
        df = eval(text)
        print(df['hs' + hs + '_name'].values[0])
        text = 'hs' + hs + '_jpn_df.query(' +"'"+ 'hs' + hs + '=="' + hs_code + '"' + "')"
        df = eval(text)
        print(df['hs' + hs + '_name'].values[0])

            
        

    #  国コード(複数) のデータを抽出　国コードは、文字列のはずだが、ときどきなる整数になるので注意
    def countries_table_create(self,countries=['105','304','103','106','601'],tables=['y_2019','year_from_1997','ym_2018_2020']):
        clist = "('" + "','".join(countries) + "')" 
        sql = """
        create table countries_{table}
        as select * from {table}
        where Country in {clist}
        """[1:-1]
        
        for table in tables:
            tg = 'drop table countries_{table}'.format(table=table)
            print(tg)
            try:
                cursor.execute(tg)
            except:
                pass
            cursor.execute(sql.format(clist=clist,table=table))

        conn.commit()
        
    # 国別折れ線グラフのときに、国与える色です。
    def national_colors(self):
        return ({'105': ['中国', 'gold'],
        '304': ['アメリカ', 'red'],
         '103': ['韓国', 'blue'],
         '106:': ['台湾', 'cyan'],
         '601:': ['オーストラリア', 'green'],
         '111:': ['タイ', 'violet'],
         '213:': ['ドイツ', 'lightgrey'],
         '110:': ['ベトナム', 'crimson'],
         '108:': ['香港', 'orangered'],
         '112:': ['シンガポール', 'aqua'],
         '147:': ['アラブ首長国連邦', 'black'],
         '137:': ['サウジ', 'darkgreen'],
         '118:': ['インドネシア', 'darkorange'],
         '113:': ['マレーシア', 'yellow'],
         '205:': ['イギリス', 'darkblue'],
         '224:': ['ロシア', 'pink'],
         '117:': ['フィリピン', 'olive'],
         '302:': ['カナダ', 'salmon'],
         '210:': ['フランス', 'indigo'],
         '305:': ['メキシコ', 'greenyellow']})
    
    # 虹の7色を割り当てる
    def rank_color(self,xlist):
        clist = ['red','ornage','yellow','green','blue','indigo','violet']
        palette = {xlist[i]:clist[i] for i in range(len(xlist))}
        return(palette)


ut = util()




# used car hs9 code
hlist= """
("870321915","870321925","870322910","870323915","870323925",
"870324910","870331100","870332915","870332925","870333910","870390100")
"""[1:-1]




# 
sql  = """
select Year,exp_imp,sum(Value) as Value
from year_from_1997
where hs9 in {hlist}
group by Year,exp_imp
"""[1:-1]

df = pd.read_sql(sql.format(hlist=hlist),conn)




import seaborn
seaborn.__version__




ut.g2(df,'Year','Value','exp_imp')




xdf = pd.DataFrame(columns={'Year':'int','Value':'int'})
#list(range(2008,2017))
for y in list(range(2008,2017)):
#for y in list(range(2008,2009)):
    db = b_dir + 'ym_custom_' + str(y) + '.db'
    xdf = pd.concat([xdf,select(db,sql.format(y=y))])




show_period(xdf.rename(columns={"Year":"period"}))




def show_country(y,head_num=10,title=""):
    x_y = y.groupby(["Country"],as_index=False )["Value"].sum()
    x_y = pd.merge(x_y,country,on='Country')
    x_sum = x_y["Value"].sum()
    x_y["percent"] = 100*(x_y["Value"]/x_sum)
    display(HTML(title))
    x= x_y.sort_values("Value",ascending=False).head(head_num)
    display(x[["Value","Country","Country_name","percent"]])
    
    c ="x.sort_values('Value').plot.barh(y=['percent'] ,x=['Country'],alpha=0.6, figsize=(12,5))"
    eval(c)




db = b_dir + 'ym_custom_2016.db'
sql = 'select Country,sum(Value) as Value from ym_custom_{y} where hs9 in ' + ulist + ' and exp_imp=1'
sql = sql + ' group by Country'
l_df = select(db,sql.format(y='2016'))




show_country(l_df)






