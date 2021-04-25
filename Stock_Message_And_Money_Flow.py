#Stock_Message_And_Money_Flow
# coding=utf-8
import tushare as ts
import pandas as pd
import jiagu
from datetime import datetime
from transformers import pipeline, AutoTokenizer,AutoModelForSequenceClassification

#在此填入你的tushare token，用户需要至少2000积分，无就找米哥
ts_token=''
#获取tushare的pro对象
def get_tushare():
    ts.set_token(ts_token)
    pro=ts.pro_api()
    return pro    
#获取最近的10个交易日的日期
def get_last_ten_trady_date(end_date):
    tushare=get_tushare()
    df = tushare.trade_cal(exchange='SSE',end_date=end_date,is_open=1)
    return str(df.tail(10)['cal_date'].values[0])
#获取最近的1个交易日的日期
def get_last_trade_date(end_date):
    tushare=get_tushare()
    df = tushare.trade_cal(exchange='SSE',end_date=end_date,is_open=1,fields=['exchange','cal_date','is_open','pretrade_date'])
    return str(df.tail(1)['pretrade_date'].values[0])
#获取对应股票指定交易日范围内的大资金流入和流出占自由流通股本比例
def get_stock_money_flow(stock_code,start_date, end_date):
    tushare=get_tushare()
    df = tushare.moneyflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
    free_share=get_stock_free_share(stock_code, end_date)
    big_buy_in_vol=(df['buy_lg_vol'].sum()+df['buy_elg_vol'].sum())*100/free_share
    big_sell_out_vol=(df['sell_lg_vol'].sum()+df['sell_elg_vol'].sum())*100/free_share
    message="股票"+stock_code+"流入占总流通盘的"+str(round(big_buy_in_vol*100,2))+"%；流出占总流通盘的"+str(round(big_sell_out_vol*100,2))+"%；净流入占总流通盘的"+str(round(big_buy_in_vol*100-big_sell_out_vol*100,2))+"%。"
    return message
#获取对应股票自由流通股本
def get_stock_free_share(stock_code,trade_date):
    tushare=get_tushare()
    df = tushare.daily_basic(ts_code=stock_code,trade_date=get_last_ten_trady_date(trade_date))
    free_share=df['free_share'].values[0]*10000
    return free_share
#获取股票对应概念
def get_stock_concept(stock_code):
    tushare=get_tushare()
    df = tushare.concept_detail(ts_code =stock_code)
    stock_concept=df['concept_name'].values.tolist()
    stock_concept.append(stock_code[0:6])
    return stock_concept
#获取主流新闻网站的快讯新闻数据
def get_market_news(trade_date):
    tushare=get_tushare()
    last_trade_date=parse_ymd(get_last_trade_date(trade_date))
    df = tushare.news(src='sina', start_date=last_trade_date+' 15:00:00', end_date=parse_ymd(trade_date)+' 23:59:59')
    return df
#将yyyymmdd格式转换成yyyy-mm-dd格式
def parse_ymd(s):
    my_date = datetime.strptime(s,'%Y%m%d')
    return str(datetime(int(my_date.year), int(my_date.month), int(my_date.day)).strftime("%Y-%m-%d"))
if __name__ == "__main__":
    stock_code_list=['300352.SZ']
    trade_date=str(datetime.now().strftime('%Y%m%d'))
    last_ten_trady_date=get_last_ten_trady_date(trade_date)
    df_market_news=get_market_news(trade_date)
    df_stock_message_and_money_flow=pd.DataFrame(columns=['tscode','message_original','message_parsed','key_word','label','score'])
    for i in range(len(stock_code_list)):
        stock_code=stock_code_list[i]
        stock_concept=get_stock_concept(stock_code)
        #加载特定股票的关键词字典
        jiagu.load_userdict(stock_concept)
        for j in range(len(df_market_news)):
            content=df_market_news.loc[j]['content']
            #中文分词
            words = jiagu.seg(content)
            key_word_flag=False
            key_words=set()
            for k in range(len(words)):
                if(stock_concept.count(words[k]))>0:
                    key_word_flag=True
                    key_words.add(words[k])
            if(key_word_flag):
                message_parsed=content+get_stock_money_flow(stock_code,last_ten_trady_date, trade_date)
                df_stock_message_and_money_flow=df_stock_message_and_money_flow.append(pd.DataFrame(data={'tscode': [stock_code], 'message_original': [content],'message_parsed':[message_parsed],'key_word':[','.join(key_words)],'label':[0],'score':[0]}),ignore_index=True)
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")
    model = AutoModelForSequenceClassification.from_pretrained("./bert-base-chinese")
    nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer,framework='pt')
    for l in range(len(df_stock_message_and_money_flow)):
        message_parsed=df_stock_message_and_money_flow.loc[l,'message_parsed']
        result=nlp(message_parsed)
        print(result[0]['label'],result[0]['score'])
        df_stock_message_and_money_flow.loc[l,'label']=result[0]['label']
        df_stock_message_and_money_flow.loc[l,'score']=result[0]['score']        
    df_stock_message_and_money_flow.to_csv('df_stock_message_and_money_flow.csv',encoding='utf_8_sig',index=False)
