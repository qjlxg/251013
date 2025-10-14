import os
import json
import time
import pandas as pd
import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import akshare as ak
import random

# --- 用户请求的修改: 数据来源配置 ---
FUND_DATA_DIR = 'fund_data'
# 确保本地数据目录存在
os.makedirs(FUND_DATA_DIR, exist_ok=True)
# ------------------------------------

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

def getURL(url, tries_num=5, sleep_time=1, time_out=15, proxies=None):
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.5, sleep_time))
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_rankings(fund_type='hh', start_date='2022-09-16', end_date='2025-09-16', proxies=None):
    periods = {
        '3y': (start_date, end_date),
        '2y': (f"{int(end_date[:4])-2}{end_date[4:]}", end_date),
        '1y': (f"{int(end_date[:4])-1}{end_date[4:]}", end_date),
        # 简单估算6个月前的日期
        '6m': (f"{int(end_date[:4])-(1 if int(end_date[5:7])<=6 else 0)}-{int(end_date[5:7])-6 if int(end_date[5:7])>6 else int(end_date[5:7])+6:02d}{end_date[7:]}", end_date),
    }

    # Fetch all fund codes and their rankings using akshare
    try:
        # 假设 ak.fund_em_open_fund_info(symbol='全部') 返回所有基金及其业绩
        df = ak.fund_em_open_fund_info(fund=fund_type, symbol='全部', start_date=periods['3y'][0].replace('-', ''), end_date=periods['3y'][1].replace('-', ''))
    except Exception as e:
        print(f"获取基金排行榜失败: {e}")
        return pd.DataFrame()
    
    # Standardize columns
    df.rename(columns={'基金代码': 'code', '基金简称': 'name'}, inplace=True)
    df.set_index('code', inplace=True)
    
    # 检查并重命名3年回报列
    if '近3年' in df.columns:
        df.rename(columns={'近3年': 'rose_3y'}, inplace=True)
    else:
        df['rose_3y'] = np.nan

    return df[['name', 'rose_3y']].copy()

def calculate_returns(df, end_date_str):
    """从净值数据计算短期回报率"""
    if df.empty:
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
    
    # 确保净值日期是datetime，并排序
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df.sort_values(by='净值日期', inplace=True)
    
    # 查找截止日期对应的净值
    end_date = pd.to_datetime(end_date_str)
    
    # 找到最接近/最新的净值作为计算基准
    latest_net_value_row = df[df['净值日期'] <= end_date].iloc[-1] if not df[df['净值日期'] <= end_date].empty else df.iloc[-1]
    latest_date = latest_net_value_row['净值日期']
    latest_net_value = latest_net_value_row['单位净值']
    
    if pd.isna(latest_net_value):
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}

    returns = {}

    # 1年回报 (rose_1y)
    date_1y_ago = latest_date - timedelta(days=365)
    # 查找1年前最接近的净值
    date_1y_value_row = df[df['净值日期'] <= date_1y_ago].iloc[-1] if not df[df['净值日期'] <= date_1y_ago].empty else None

    if date_1y_value_row is not None:
        net_value_1y_ago = date_1y_value_row['单位净值']
        returns['rose_1y'] = (latest_net_value / net_value_1y_ago - 1) * 100
    else:
        returns['rose_1y'] = np.nan

    # 6个月回报 (rose_6m)
    date_6m_ago = latest_date - timedelta(days=180) # Approximation
    # 查找6个月前最接近的净值
    date_6m_value_row = df[df['净值日期'] <= date_6m_ago].iloc[-1] if not df[df['净值日期'] <= date_6m_ago].empty else None

    if date_6m_value_row is not None:
        net_value_6m_ago = date_6m_value_row['单位净值']
        returns['rose_6m'] = (latest_net_value / net_value_6m_ago - 1) * 100
    else:
        returns['rose_6m'] = np.nan
        
    returns['csv_filename'] = 'Local_or_Downloaded_Data' # Indicator for successful retrieval/processing

    return returns

def download_fund_csv(fund_code, start_date='20200101', end_date=None, proxies=None):
    """
    修改后的基金净值数据获取函数。
    优先从本地 fund_data/{fund_code}.csv 加载数据，如果不存在或读取失败，则从网上下载。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    local_path = os.path.join(FUND_DATA_DIR, f'{fund_code}.csv')
    df_fund = pd.DataFrame()
    
    # 1. 尝试从本地加载
    try:
        if os.path.exists(local_path):
            df_fund = pd.read_csv(local_path)
            # 统一列名以匹配计算逻辑 (date -> 净值日期, net_value -> 单位净值)
            if 'date' in df_fund.columns and 'net_value' in df_fund.columns:
                df_fund.rename(columns={'date': '净值日期', 'net_value': '单位净值'}, inplace=True)
                
                # 简单检查数据是否有效
                if not df_fund.empty and '单位净值' in df_fund.columns:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功从本地加载 {local_path}")
                else:
                    raise ValueError("本地数据为空或格式不正确，尝试下载更新。")
            else:
                raise ValueError("本地CSV文件格式不正确，尝试下载。")
            
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 从本地加载基金 {fund_code} 净值失败: {e}。将尝试从网上下载。")
        df_fund = pd.DataFrame() # 确保清空数据框

    # 2. 尝试从网上下载 (只有当本地加载失败时)
    if df_fund.empty:
        try:
            # 使用 akshare 获取净值数据
            df_fund = ak.fund_em_open_fund_info(fund=fund_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''), indicator='单位净值走势')
            if df_fund.empty:
                raise ValueError("在线下载数据为空。")

            # 统一列名 (akshare 通常返回 '净值日期' 和 '单位净值')
            if '净值日期' not in df_fund.columns or '单位净值' not in df_fund.columns:
                # 尝试重命名，防止akshare接口变动
                if len(df_fund.columns) >= 2:
                    df_fund.columns = ['净值日期', '单位净值'] + list(df_fund.columns[2:]) 

            # 确保日期和净值是正确的类型
            df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期']).dt.strftime('%Y-%m-%d')
            df_fund['单位净值'] = pd.to_numeric(df_fund['单位净值'], errors='coerce')
            
            # 仅保留日期和净值列
            df_fund = df_fund[['净值日期', '单位净值']].dropna(subset=['单位净值'])

            # 将下载的数据保存到本地，以便下次使用 (date,net_value 格式)
            df_fund_save = df_fund.copy()
            df_fund_save.rename(columns={'净值日期': 'date', '单位净值': 'net_value'}, inplace=True)
            df_fund_save.to_csv(local_path, index=False, encoding='utf-8')
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功下载并保存基金 {fund_code} 净值到 {local_path}")
            
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 下载基金 {fund_code} 净值失败: {e}")
            return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
            
    # 3. 计算短期回报
    if not df_fund.empty:
        # 对净值数据进行时间过滤
        df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期'])
        df_fund = df_fund[(df_fund['净值日期'] >= pd.to_datetime(start_date)) & 
                          (df_fund['净值日期'] <= pd.to_datetime(end_date))]
        
        returns = calculate_returns(df_fund, end_date)
        return returns
    else:
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}


def get_fund_details(fund_code, proxies=None):
    """从网页爬取基金规模和基金经理 (保持原有逻辑)"""
    url = f"http://fund.eastmoney.com/{fund_code}.html"
    res = getURL(url, proxies=proxies)
    details = {'scale': np.nan, 'manager': 'N/A'}

    if res:
        soup = BeautifulSoup(res.text, 'html.parser')

        # 1. 基金规模 (scale)
        try:
            scale_tag = soup.find('td', text=re.compile(r'基金规模'))
            if scale_tag:
                scale_value_tag = scale_tag.find_next_sibling('td')
                if scale_value_tag:
                    scale_text = scale_value_tag.text.strip()
                    match = re.search(r'([\d.]+)([亿万]元)', scale_text)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2)
                        # 统一转换为亿元
                        if '万元' in unit:
                            value /= 10000 
                        details['scale'] = round(value, 2)
        except Exception as e:
            print(f"获取基金 {fund_code} 规模失败: {e}")

        # 2. 基金经理 (manager)
        try:
            manager_tag = soup.find('a', attrs={'href': re.compile(r'/manager/\d+\.html')})
            if manager_tag:
                details['manager'] = manager_tag.text.strip()
        except Exception as e:
            print(f"获取基金 {fund_code} 经理失败: {e}")

    return details

def analyze_fund(fund_code, start_date, end_date):
    """
    计算基金的夏普比率和最大回撤。
    已修改为优先从本地加载数据。
    """
    local_path = os.path.join(FUND_DATA_DIR, f'{fund_code}.csv')
    df_fund = pd.DataFrame()
    
    # 优先从本地加载
    try:
        if os.path.exists(local_path):
            df_fund = pd.read_csv(local_path)
            if 'date' in df_fund.columns and 'net_value' in df_fund.columns:
                df_fund.rename(columns={'date': '净值日期', 'net_value': '单位净值'}, inplace=True)
                df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期'])
                df_fund['单位净值'] = pd.to_numeric(df_fund['单位净值'], errors='coerce')
                df_fund = df_fund.dropna(subset=['单位净值'])
                df_fund.set_index('净值日期', inplace=True)
                df_fund = df_fund.sort_index()
                
                # 过滤日期
                df_fund = df_fund[df_fund.index >= pd.to_datetime(start_date)]
                df_fund = df_fund[df_fund.index <= pd.to_datetime(end_date)]
                
                if not df_fund.empty:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功从本地加载 {local_path} 进行风险分析")
                else:
                    raise ValueError("本地数据不在指定日期范围内，尝试下载。")
            else:
                raise ValueError("本地CSV文件格式不正确，尝试下载。")
        else:
            raise FileNotFoundError("本地CSV文件不存在，尝试下载。")
    except Exception:
        # Fallback to online download
        try:
            df_fund = ak.fund_em_open_fund_info(
                fund=fund_code, 
                start_date=start_date.replace('-', ''), 
                end_date=end_date.replace('-', ''), 
                indicator='单位净值走势'
            )
            # 假设 akshare 数据的列名为 '净值日期' 和 '单位净值'
            df_fund.columns = ['净值日期', '单位净值', '累计净值'] 
            df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期'])
            df_fund['单位净值'] = pd.to_numeric(df_fund['单位净值'], errors='coerce')
            df_fund = df_fund.dropna(subset=['单位净值'])
            df_fund.set_index('净值日期', inplace=True)
            df_fund = df_fund.sort_index()
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功下载基金 {fund_code} 净值进行风险分析")
        except Exception as e:
            return {'error': f"无法获取基金 {fund_code} 净值进行风险分析: {e}"}

    if df_fund.empty:
        return {'error': f"基金 {fund_code} 在 {start_date} 到 {end_date} 期间没有有效净值数据"}

    net_values = df_fund['单位净值']

    # 1. 计算每日收益率
    daily_returns = net_values.pct_change().dropna()

    if daily_returns.empty:
        return {'error': f"基金 {fund_code} 每日收益率计算失败"}

    # 2. 夏普比率 (Sharpe Ratio)
    risk_free_rate = 0 # 简化，无风险利率设为0
    annual_trading_days = 250
    
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    if std_daily_return == 0:
        sharpe_ratio = np.nan
    else:
        # 年化夏普比率
        annualized_mean_excess_return = (mean_daily_return - (risk_free_rate / annual_trading_days)) * annual_trading_days
        annualized_std = std_daily_return * np.sqrt(annual_trading_days)
        sharpe_ratio = annualized_mean_excess_return / annualized_std
        
    # 3. 最大回撤 (Max Drawdown)
    cumulative_returns = (1 + daily_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * -1 

    return {
        'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
        'max_drawdown': round(max_drawdown * 100, 2) # 百分比形式
    }


def get_all_fund_data(fund_type='hh', start_date='2022-09-16', end_date='2025-09-16'):
    """
    获取基金数据的主函数。
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始获取基金数据...")
    
    # 1. 获取基金排名/基础列表
    ranking_data = get_fund_rankings(fund_type, start_date, end_date)
    if ranking_data.empty:
        print("获取基金列表失败，请检查网络连接或akshare接口。")
        return pd.DataFrame()
        
    fund_codes = ranking_data.index.tolist()
    merged_data = ranking_data.copy()
    
    # 2. 初始化额外的列
    merged_data['rose_1y'] = np.nan
    merged_data['rose_6m'] = np.nan
    merged_data['scale'] = np.nan
    merged_data['manager'] = 'N/A'
    merged_data['sharpe_ratio'] = np.nan
    merged_data['max_drawdown'] = np.nan
    
    # 3. 逐个基金获取详细数据、短期回报和风险指标
    for i, fund_code in enumerate(fund_codes, 1):
        print(f"[{i}/{len(fund_codes)}] 处理基金 {fund_code}...")
        
        # 下载历史净值并计算短期回报 (已修改为优先本地加载)
        # start_date for download is set to a long historical date '2020-01-01'
        download_start_date = '2020-01-01'
        result = download_fund_csv(fund_code, start_date=download_start_date, end_date=end_date)
        if result['csv_filename']:
            merged_data.loc[merged_data.index == fund_code, 'rose_1y'] = result['rose_1y']
            merged_data.loc[merged_data.index == fund_code, 'rose_6m'] = result['rose_6m']
            
        # 获取基金详情
        details = get_fund_details(fund_code)
        merged_data.loc[merged_data.index == fund_code, 'scale'] = details.get('scale', np.nan)
        merged_data.loc[merged_data.index == fund_code, 'manager'] = details.get('manager', 'N/A')
        
        # 获取并分析风险指标 (已修改为优先本地加载)
        risk_metrics = analyze_fund(fund_code, start_date, end_date) 
        if 'error' not in risk_metrics:
            merged_data.loc[merged_data.index == fund_code, 'sharpe_ratio'] = risk_metrics['sharpe_ratio']
            merged_data.loc[merged_data.index == fund_code, 'max_drawdown'] = risk_metrics['max_drawdown']
            
        # 避免过于频繁的请求
        time.sleep(random.uniform(0.1, 0.5)) 

    # 4. 数据清洗和排序
    merged_data.dropna(subset=['rose_3y', 'rose_1y', 'sharpe_ratio'], how='all', inplace=True)
    
    merged_data.sort_values(by=['sharpe_ratio', 'rose_3y'], ascending=[False, False], inplace=True)
    
    # 格式化输出
    merged_data['rose_3y'] = merged_data['rose_3y'].round(2).astype(str) + '%'
    merged_data['rose_1y'] = merged_data['rose_1y'].round(2).astype(str) + '%'
    merged_data['rose_6m'] = merged_data['rose_6m'].round(2).astype(str) + '%'
    merged_data['max_drawdown'] = merged_data['max_drawdown'].round(2).astype(str) + '%'
    
    # 调整列顺序
    final_columns = [
        'name', 'manager', 'scale', 
        'rose_3y', 'rose_1y', 'rose_6m', 
        'sharpe_ratio', 'max_drawdown'
    ]
    
    merged_data = merged_data.reindex(columns=final_columns)
    
    return merged_data

# --- Main execution block ---
if __name__ == '__main__':
    # 获取日期，默认是今天
    end_date = datetime.now().strftime('%Y-%m-%d')
    # 筛选的开始日期（用于风险指标计算，例如2年）
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

    # 示例: 获取混合型(hh)基金数据
    df_result = get_all_fund_data(fund_type='hh', start_date=start_date, end_date=end_date)
    
    if not df_result.empty:
        # 输出到 Excel
        output_filename = f"fund_screener_result_{end_date.replace('-', '')}.xlsx"
        df_result.to_excel(output_filename, index=True, encoding='utf-8')
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 筛选结果已保存到 {output_filename}")
        print("\n--- 筛选结果预览 ---")
        print(df_result.head(20))
    else:
        print("\n未找到符合条件的基金数据。")
