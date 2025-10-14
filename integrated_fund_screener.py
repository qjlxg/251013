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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 全局配置与常量 ---
FUND_DATA_DIR = 'fund_data'
# 用户建议：配置无风险利率 (年化百分比，例如 3% -> 0.03)
RISK_FREE_RATE = 0.03 
ANNUAL_TRADING_DAYS = 250
MAX_WORKERS = 10 # 线程池最大工作线程数
# ----------------------

# 确保本地数据目录存在
os.makedirs(FUND_DATA_DIR, exist_ok=True)

def randHeader():
    """生成随机的请求头"""
    head_user_agent = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:51.0) Gecko/20100101 Firefox/51.0',
    ]
    return {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'User-Agent': random.choice(head_user_agent),
        'Referer': 'http://fund.eastmoney.com/'
    }

def getURL(url, tries_num=5, sleep_time=1, time_out=15, proxies=None):
    """请求URL并处理重试"""
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.1, sleep_time))
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            res.raise_for_status()
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            # 缩短重试间隔，因为是并行操作，不应阻塞太久
            time.sleep(random.uniform(1, 2) + i * 2) 
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_rankings(fund_type='hh', start_date='2022-09-16', end_date='2025-09-16'):
    """
    获取基金排行榜，并加入数据完整性检查。
    """
    periods = {
        '3y': (start_date, end_date),
    }

    try:
        # 使用 akshare 获取所有基金及其业绩
        # 注意: akshare 接口可能只返回部分日期指标，我们主要依赖它获取完整的基金代码列表和3年回报
        df = ak.fund_em_open_fund_info(fund=fund_type, symbol='全部', 
                                       start_date=periods['3y'][0].replace('-', ''), 
                                       end_date=periods['3y'][1].replace('-', ''))
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
        
    # --- 数据完整性增强：过滤掉3年回报为空的基金 ---
    initial_count = len(df)
    df.replace('-', np.nan, inplace=True) # 将 '-' 替换为 NaN
    df['rose_3y'] = pd.to_numeric(df['rose_3y'], errors='coerce')
    df.dropna(subset=['rose_3y'], inplace=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始获取基金数量: {initial_count}, 过滤后剩余: {len(df)}")
    # ---------------------------------------------

    return df[['name', 'rose_3y']].copy()

def calculate_returns(df, end_date_str):
    """从净值数据计算短期回报率 (1年, 6个月)"""
    if df.empty or '单位净值' not in df.columns:
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
    
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df.sort_values(by='净值日期', inplace=True)
    
    end_date = pd.to_datetime(end_date_str)
    
    latest_net_value_row = df[df['净值日期'] <= end_date].iloc[-1] if not df[df['净值日期'] <= end_date].empty else None
    
    if latest_net_value_row is None:
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
        
    latest_date = latest_net_value_row['净值日期']
    latest_net_value = latest_net_value_row['单位净值']
    
    returns = {}

    # 1年回报 (rose_1y)
    date_1y_ago = latest_date - timedelta(days=365)
    date_1y_value_row = df[df['净值日期'] <= date_1y_ago].iloc[-1] if not df[df['净值日期'] <= date_1y_ago].empty else None

    if date_1y_value_row is not None:
        net_value_1y_ago = date_1y_value_row['单位净值']
        returns['rose_1y'] = (latest_net_value / net_value_1y_ago - 1) * 100
    else:
        returns['rose_1y'] = np.nan

    # 6个月回报 (rose_6m)
    date_6m_ago = latest_date - timedelta(days=180) 
    date_6m_value_row = df[df['净值日期'] <= date_6m_ago].iloc[-1] if not df[df['净值日期'] <= date_6m_ago].empty else None

    if date_6m_value_row is not None:
        net_value_6m_ago = date_6m_value_row['单位净值']
        returns['rose_6m'] = (latest_net_value / net_value_6m_ago - 1) * 100
    else:
        returns['rose_6m'] = np.nan
        
    returns['csv_filename'] = 'Local_or_Downloaded_Data' 
    return returns

def download_fund_csv(fund_code, start_date='2020-01-01', end_date=None):
    """
    优先从本地 fund_data/{fund_code}.csv 加载数据，失败则回退到网上下载并保存。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    local_path = os.path.join(FUND_DATA_DIR, f'{fund_code}.csv')
    df_fund = pd.DataFrame()
    
    # 1. 尝试从本地加载
    try:
        if os.path.exists(local_path):
            df_fund = pd.read_csv(local_path)
            if 'date' in df_fund.columns and 'net_value' in df_fund.columns:
                df_fund.rename(columns={'date': '净值日期', 'net_value': '单位净值'}, inplace=True)
                if not df_fund.empty and '单位净值' in df_fund.columns:
                    # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功从本地加载 {local_path}")
                    pass
                else:
                    raise ValueError("本地数据为空或格式不正确，尝试下载更新。")
            else:
                raise ValueError("本地CSV文件格式不正确，尝试下载。")
            
    except Exception:
        # 尝试从网上下载
        try:
            df_fund = ak.fund_em_open_fund_info(fund=fund_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''), indicator='单位净值走势')
            if df_fund.empty:
                raise ValueError("在线下载数据为空。")

            # 统一列名
            if '净值日期' not in df_fund.columns or '单位净值' not in df_fund.columns:
                if len(df_fund.columns) >= 2:
                    df_fund.columns = ['净值日期', '单位净值'] + list(df_fund.columns[2:]) 

            df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期']).dt.strftime('%Y-%m-%d')
            df_fund['单位净值'] = pd.to_numeric(df_fund['单位净值'], errors='coerce')
            df_fund = df_fund[['净值日期', '单位净值']].dropna(subset=['单位净值'])

            # 保存到本地 (date,net_value 格式)
            df_fund_save = df_fund.copy()
            df_fund_save.rename(columns={'净值日期': 'date', '单位净值': 'net_value'}, inplace=True)
            df_fund_save.to_csv(local_path, index=False, encoding='utf-8')
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 下载并保存基金 {fund_code} 净值到 {local_path}")
            
        except Exception as e:
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 下载基金 {fund_code} 净值失败: {e}")
            return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}
            
    # 2. 计算短期回报
    if not df_fund.empty:
        df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期'])
        df_fund = df_fund[(df_fund['净值日期'] >= pd.to_datetime(start_date)) & 
                          (df_fund['净值日期'] <= pd.to_datetime(end_date))]
        
        returns = calculate_returns(df_fund, end_date)
        return returns
    else:
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}


def get_fund_details(fund_code, cache={}):
    """从网页爬取基金规模和基金经理，并使用局部缓存"""
    if fund_code in cache:
        return cache[fund_code]

    url = f"http://fund.eastmoney.com/{fund_code}.html"
    res = getURL(url)
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
        except Exception:
            pass # 忽略爬取失败
        # 2. 基金经理 (manager)
        try:
            manager_tag = soup.find('a', attrs={'href': re.compile(r'/manager/\d+\.html')})
            if manager_tag:
                details['manager'] = manager_tag.text.strip()
        except Exception:
            pass # 忽略爬取失败
            
    cache[fund_code] = details # 缓存结果
    return details

def analyze_fund(fund_code, start_date, end_date):
    """
    计算基金的夏普比率、波动率和最大回撤。
    已修改为优先从本地加载数据，并使用配置的无风险利率。
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
                
                df_fund = df_fund[df_fund.index >= pd.to_datetime(start_date)]
                df_fund = df_fund[df_fund.index <= pd.to_datetime(end_date)]
                
                if df_fund.empty:
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
            df_fund.columns = ['净值日期', '单位净值', '累计净值'] 
            df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期'])
            df_fund['单位净值'] = pd.to_numeric(df_fund['单位净值'], errors='coerce')
            df_fund = df_fund.dropna(subset=['单位净值'])
            df_fund.set_index('净值日期', inplace=True)
            df_fund = df_fund.sort_index()
        except Exception as e:
            return {'error': f"无法获取基金 {fund_code} 净值进行风险分析"}

    if df_fund.empty:
        return {'error': f"基金 {fund_code} 在 {start_date} 到 {end_date} 期间没有有效净值数据"}

    net_values = df_fund['单位净值']

    # 1. 每日收益率
    daily_returns = net_values.pct_change().dropna()

    if daily_returns.empty:
        return {'error': f"基金 {fund_code} 每日收益率计算失败"}

    # 2. 波动率 (Volatitlity - 年化标准差)
    std_daily_return = daily_returns.std()
    annualized_volatility = std_daily_return * np.sqrt(ANNUAL_TRADING_DAYS)
    
    # 3. 夏普比率 (Sharpe Ratio)
    mean_daily_return = daily_returns.mean()
    
    # 年化夏普比率
    risk_free_rate_daily = RISK_FREE_RATE / ANNUAL_TRADING_DAYS
    
    if std_daily_return == 0:
        sharpe_ratio = np.nan
    else:
        annualized_mean_excess_return = (mean_daily_return - risk_free_rate_daily) * ANNUAL_TRADING_DAYS
        sharpe_ratio = annualized_mean_excess_return / annualized_volatility
        
    # 4. 最大回撤 (Max Drawdown)
    cumulative_returns = (1 + daily_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * -1 

    return {
        'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
        'max_drawdown': round(max_drawdown * 100, 2), # 百分比形式
        'volatility': round(annualized_volatility * 100, 2) # 百分比形式
    }

def process_single_fund(fund_code, fund_name, start_date, end_date):
    """
    单个基金的并行处理逻辑。
    返回包含所有指标的字典或None。
    """
    result_data = {
        'code': fund_code,
        'name': fund_name,
        'rose_1y': np.nan, 
        'rose_6m': np.nan, 
        'scale': np.nan, 
        'manager': 'N/A', 
        'sharpe_ratio': np.nan, 
        'max_drawdown': np.nan,
        'volatility': np.nan,
        'error': None
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] 正在处理基金 {fund_code} - {fund_name}...")

    # 1. 下载历史净值并计算短期回报
    download_start_date = '2020-01-01'
    returns = download_fund_csv(fund_code, start_date=download_start_date, end_date=end_date)
    if returns['csv_filename']:
        result_data['rose_1y'] = returns['rose_1y']
        result_data['rose_6m'] = returns['rose_6m']
    
    # 2. 获取基金详情
    # 注意：get_fund_details 内部已实现对重复URL的局部缓存
    details = get_fund_details(fund_code)
    result_data['scale'] = details.get('scale', np.nan)
    result_data['manager'] = details.get('manager', 'N/A')
    
    # 3. 获取并分析风险指标
    risk_metrics = analyze_fund(fund_code, start_date, end_date) 
    if 'error' not in risk_metrics:
        result_data['sharpe_ratio'] = risk_metrics['sharpe_ratio']
        result_data['max_drawdown'] = risk_metrics['max_drawdown']
        result_data['volatility'] = risk_metrics['volatility']
    else:
        result_data['error'] = risk_metrics['error']
        print(f"[{time.strftime('%H:%M:%S')}] 基金 {fund_code} 风险分析失败: {risk_metrics['error']}")

    return result_data

def get_all_fund_data(fund_type='hh', start_date='2023-09-16', end_date='2025-09-16', limit=100):
    """
    获取基金数据的主函数，使用多线程加速。
    """
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始获取和分析基金数据...")
    print(f"分析时间范围: {start_date} 至 {end_date}, 无风险利率: {RISK_FREE_RATE*100}%")
    
    # 1. 获取基金排名/基础列表
    ranking_data = get_fund_rankings(fund_type, start_date, end_date)
    if ranking_data.empty:
        print("获取基金列表失败，请检查网络连接或akshare接口。")
        return pd.DataFrame()
        
    fund_list_to_process = ranking_data.head(limit).index.tolist()
    initial_ranking_data = ranking_data.head(limit).copy()
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 准备处理前 {len(fund_list_to_process)} 只基金...")
    
    final_results = []
    
    # 2. 使用多线程并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_fund, code, initial_ranking_data.loc[code, 'name'], start_date, end_date): code for code in fund_list_to_process}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
                print(f"[{time.strftime('%H:%M:%S')}] 完成处理 {i}/{len(fund_list_to_process)} 基金。")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 基金处理过程中发生错误: {e}")
                
    
    if not final_results:
        print("所有基金处理失败或无有效结果。")
        return pd.DataFrame()

    # 3. 数据整合、清洗和排序
    results_df = pd.DataFrame(final_results).set_index('code')
    
    # 合并3年回报率
    merged_data = initial_ranking_data.merge(results_df.drop(columns=['name', 'error'], errors='ignore'), left_index=True, right_index=True, how='left')

    # 清洗：移除没有成功计算任何关键指标的行
    # 至少要有3年回报或者夏普比率
    merged_data.dropna(subset=['rose_3y', 'sharpe_ratio'], how='all', inplace=True)
    
    # 排序：夏普比率降序，3年回报降序
    merged_data.sort_values(by=['sharpe_ratio', 'rose_3y'], ascending=[False, False], inplace=True)
    
    # 格式化输出
    for col in ['rose_3y', 'rose_1y', 'rose_6m', 'max_drawdown', 'volatility']:
        merged_data[col] = merged_data[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
        
    merged_data['sharpe_ratio'] = merged_data['sharpe_ratio'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
    
    # 调整列顺序
    final_columns = [
        'name', 'manager', 'scale', 
        'rose_3y', 'rose_1y', 'rose_6m', 
        'sharpe_ratio', 'volatility', 'max_drawdown'
    ]
    
    merged_data = merged_data.reindex(columns=final_columns)
    
    end_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 总运行时间: {end_time - start_time:.2f} 秒。")
    
    return merged_data

# --- Main execution block ---
if __name__ == '__main__':
    # 配置参数
    end_date = datetime.now().strftime('%Y-%m-%d')
    # 风险指标计算基于最近2年数据
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    # 限制处理前100只基金
    process_limit = 100 
    
    # 示例: 获取混合型(hh)基金数据
    df_result = get_all_fund_data(
        fund_type='hh', 
        start_date=start_date, 
        end_date=end_date,
        limit=process_limit
    )
    
    if not df_result.empty:
        # 输出到 Excel
        output_filename = f"fund_screener_optimized_result_{end_date.replace('-', '')}.xlsx"
        df_result.to_excel(output_filename, index=True, index_label='代码', encoding='utf-8')
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 筛选结果已保存到 {output_filename}")
        print("\n--- 筛选结果预览 (前20条) ---")
        print(df_result.head(20).to_markdown())
    else:
        print("\n未找到符合条件的基金数据。")
