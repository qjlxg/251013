import os
import json
import time
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random

def randHeader():
    head_user_agent = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
    ]
    return {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'User-Agent': random.choice(head_user_agent),
        'Referer': 'http://fund.eastmoney.com/'
    }

def getURL(url, tries_num=5, sleep_time=1, time_out=15):
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))
            res = requests.get(url, headers=randHeader(), timeout=time_out)
            res.raise_for_status()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_rankings(fund_type='hh', start_date='2022-09-16', end_date='2025-09-16'):
    periods = {
        '3y': (start_date, end_date),
        '2y': (f"{int(end_date[:4])-2}{end_date[4:]}", end_date),
        '1y': (f"{int(end_date[:4])-1}{end_date[4:]}", end_date),
        '6m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=6 else 0)}-{int(end_date[5:7])-6:02d}{end_date[7:]}", end_date),
        '3m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=3 else 0)}-{int(end_date[5:7])-3:02d}{end_date[7:]}", end_date)
    }
    fund_data_dir = 'fund_data'
    fund_codes = [f[:-4] for f in os.listdir(fund_data_dir) if f.endswith('.csv')]
    all_data = []
    for period, (sd_str, ed_str) in periods.items():
        sd = datetime.strptime(sd_str, '%Y-%m-%d')
        ed = datetime.strptime(ed_str, '%Y-%m-%d')
        roses = []
        for code in fund_codes:
            try:
                df = pd.read_csv(os.path.join(fund_data_dir, f'{code}.csv'))
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').drop_duplicates('date', keep='last')
                df_ed = df[df['date'] <= ed]
                df_sd = df[df['date'] >= sd]
                if df_ed.empty or df_sd.empty:
                    rose = np.nan
                else:
                    net_ed = df_ed['net_value'].iloc[-1]
                    net_sd = df_sd['net_value'].iloc[0]
                    rose = (net_ed / net_sd - 1)
                roses.append(rose)
            except Exception as e:
                print(f"计算 {code} {period} rose 失败: {e}")
                roses.append(np.nan)
        # 排除 NaN
        valid_idx = [i for i, r in enumerate(roses) if not np.isnan(r)]
        valid_roses = [roses[i] for i in valid_idx]
        valid_codes = [fund_codes[i] for i in valid_idx]
        if not valid_roses:
            print(f"{period} 无有效数据")
            continue
        # 排序（desc）
        sorted_idx = np.argsort([-r for r in valid_roses])
        ranks = np.empty(len(valid_roses))
        ranks[sorted_idx] = np.arange(1, len(valid_roses) + 1)
        rank_rs = ranks / len(valid_roses)
        df = pd.DataFrame({
            'code': valid_codes,
            f'rose({period})': valid_roses,
            f'rank({period})': ranks,
            f'rank_r({period})': rank_rs
        })
        df.set_index('code', inplace=True)
        all_data.append(df)
        print(f"获取 {period} 排名数据：{len(df)} 条（总计 {len(fund_codes)}）")
    if all_data:
        df_final = all_data[0].copy()
        for df in all_data[1:]:
            df_final = df_final.join(df, how='outer')
        # 临时 name
        df_final['name'] = df_final.index + 'C'
        df_final.to_csv('fund_rankings.csv', encoding='gbk')
        print(f"排名数据已保存至 'fund_rankings.csv'")
        return df_final
    return pd.DataFrame()

def apply_4433_rule(df, total_records):
    thresholds = {
        '3y': 0.25, '2y': 0.25, '1y': 0.25,
        '6m': 1/3, '3m': 1/3
    }
    filtered_df = df.copy()
    for period in thresholds:
        rank_col = f'rank_r({period})'
        if rank_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[rank_col] <= thresholds[period]]
    print(f"四四三三法则筛选出 {len(filtered_df)} 只基金")
    return filtered_df

def download_fund_csv(fund_code: str, start_date: str = '20200101', end_date: str = None) -> dict:
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    csv_filename = f'fund_data/{fund_code}.csv'
    if not os.path.exists(csv_filename):
        print(f"基金 {fund_code} 本地文件不存在")
        with open('failed_funds.txt', 'a') as f:
            f.write(f"{fund_code}: 文件不存在\n")
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
    try:
        df = pd.read_csv(csv_filename)
        df['净值日期'] = pd.to_datetime(df['date'])
        df['单位净值'] = pd.to_numeric(df['net_value'], errors='coerce')
        df['累计净值'] = ''  # 无数据，置空
        df['日增长率'] = ''  # 无数据，置空
        one_year_ago = datetime.now() - timedelta(days=365)
        six_month_ago = datetime.now() - timedelta(days=182)
        recent_data = df[df['净值日期'] >= one_year_ago]
        six_month_data = df[df['净值日期'] >= six_month_ago]
        rose_1y = (recent_data['单位净值'].iloc[-1] / recent_data['单位净值'].iloc[0] - 1) * 100 if len(recent_data) > 1 else np.nan
        rose_6m = (six_month_data['单位净值'].iloc[-1] / six_month_data['单位净值'].iloc[0] - 1) * 100 if len(six_month_data) > 1 else np.nan
        return {'csv_filename': csv_filename, 'rose_1y': rose_1y, 'rose_6m': rose_6m}
    except Exception as e:
        print(f"基金 {fund_code} 读取失败: {e}")
        with open('failed_funds.txt', 'a') as f:
            f.write(f"{fund_code}: {str(e)}\n")
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}

def get_fund_details(fund_code):
    try:
        url = f'http://fund.eastmoney.com/f10/{fund_code}.html'
        response = getURL(url)
        if not response:
            raise ValueError("无法获取响应")
        tables = pd.read_html(response.text)
        if len(tables) < 2:
            raise ValueError("表格数量不足")
        df = tables[0]
        result = {
            'fund_code': fund_code,
            'fund_name': df.get('基金全称', ['N/A'])[0],
            'fund_type': df.get('基金类型', ['N/A'])[0],
            'scale': float(df.get('基金规模', ['0'])[0].replace('亿元', '')) if '亿元' in df.get('基金规模', ['0'])[0] else 0,
            'manager': df.get('基金经理', ['N/A'])[0]
        }
        return result
    except Exception as e:
        print(f"获取 {fund_code} 详情失败: {e}")
        return {'fund_code': fund_code, 'fund_name': 'N/A', 'fund_type': 'N/A', 'scale': 0, 'manager': 'N/A'}

def get_fund_managers(fund_code, output_dir='data'):
    try:
        url = f'http://fundf10.eastmoney.com/jjjl_{fund_code}.html'
        response = getURL(url)
        if not response:
            raise ValueError("无法获取响应")
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', class_='fjjl')
        if not table:
            raise ValueError("未找到经理表格")
        rows = table.find_all('tr')[1:]
        result = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                result.append({
                    'name': cols[2].text.strip(),
                    'tenure_start': cols[3].text.strip(),
                    'return': float(cols[4].text.strip().replace('%', '')) if '%' in cols[4].text else np.nan
                })
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"fund_managers_{fund_code}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"基金经理数据已保存至 '{output_path}'")
        return result
    except Exception as e:
        print(f"获取基金经理数据失败: {e}")
        return []

def analyze_fund(fund_code, start_date, end_date):
    try:
        csv_filename = f'fund_data/{fund_code}.csv'
        if not os.path.exists(csv_filename):
            raise ValueError("本地文件不存在")
        df = pd.read_csv(csv_filename)
        df['净值日期'] = pd.to_datetime(df['date'])
        df['单位净值'] = pd.to_numeric(df['net_value'], errors='coerce')
        returns = df['单位净值'].pct_change().dropna()
        if returns.empty:
            raise ValueError("没有足够的回报数据")
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_returns - 0.03) / annual_volatility if annual_volatility != 0 else 0
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        result = {
            'fund_code': fund_code,
            'annual_returns': float(annual_returns),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }
        output_path = f'data/risk_metrics_{fund_code}.json'
        os.makedirs('data', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"风险指标数据已保存至 '{output_path}'")
        return result
    except Exception as e:
        print(f"分析基金 {fund_code} 风险参数失败: {e}")
        return {"error": "风险参数计算失败"}

def main_scraper():
    print("开始从本地 fund_data 读取并计算基金排名并筛选...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    rankings_df = get_fund_rankings(fund_type='hh', start_date=start_date, end_date=end_date)
    
    if not rankings_df.empty:
        total_records = len(rankings_df)
        recommended_df = apply_4433_rule(rankings_df, total_records)
        # 更新真实 name（从网上获取）
        for code in recommended_df.index:
            details = get_fund_details(code)
            recommended_df.loc[code, 'name'] = details.get('fund_name', code + 'C')
        recommended_df['类型'] = recommended_df['name'].apply(
            lambda x: '混合型' if '混合' in x else '股票型' if '股票' in x else '指数型' if '指数' in x else '未知'
        )
        recommended_path = 'recommended_cn_funds.csv'
        recommended_df.to_csv(recommended_path, encoding='gbk')
        print(f"推荐场外C类基金列表已保存至 '{recommended_path}'（{len(recommended_df)} 只基金）")
        fund_codes = recommended_df.index.tolist()
    else:
        print("排名数据为空，退出")
        return
    
    merged_data = recommended_df.copy()
    merged_data['rose_1y'] = np.nan
    merged_data['rose_6m'] = np.nan
    merged_data['scale'] = np.nan
    merged_data['manager'] = 'N/A'
    merged_data['sharpe_ratio'] = np.nan
    merged_data['max_drawdown'] = np.nan
    
    for i, fund_code in enumerate(fund_codes, 1):
        print(f"[{i}/{len(fund_codes)}] 处理基金 {fund_code}...")
        # 读取本地历史净值并计算短期回报
        result = download_fund_csv(fund_code, start_date='20200101', end_date=end_date)
        if result['csv_filename']:
            merged_data.loc[merged_data.index == fund_code, 'rose_1y'] = result['rose_1y']
            merged_data.loc[merged_data.index == fund_code, 'rose_6m'] = result['rose_6m']
        # 获取基金详情（name 已更新，但这里更新 scale 和 manager）
        details = get_fund_details(fund_code)
        merged_data.loc[merged_data.index == fund_code, 'scale'] = details.get('scale', 0)
        merged_data.loc[merged_data.index == fund_code, 'manager'] = details.get('manager', 'N/A')
        # 获取并分析风险指标
        risk_metrics = analyze_fund(fund_code, start_date, end_date)
        if 'error' not in risk_metrics:
            merged_data.loc[merged_data.index == fund_code, 'sharpe_ratio'] = risk_metrics.get('sharpe_ratio', np.nan)
            merged_data.loc[merged_data.index == fund_code, 'max_drawdown'] = risk_metrics.get('max_drawdown', np.nan)
        # 获取经理数据
        get_fund_managers(fund_code)
        time.sleep(5)
    
    merged_path = 'merged_funds.csv'
    merged_data.to_csv(merged_path, encoding='gbk')
    print(f"合并数据（含1年/6月回报、规模等）保存至 '{merged_path}'")

if __name__ == '__main__':
    main_scraper()
