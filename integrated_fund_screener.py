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
RISK_FREE_RATE = 0.03 
ANNUAL_TRADING_DAYS = 250
MAX_WORKERS = 10 
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
            return res
        except requests.RequestException as e:
            time.sleep(random.uniform(1, 2) + i * 2) 
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_local_fund_list():
    """
    修改后的函数：不再爬取排行榜。
    通过扫描本地 fund_data 目录中的 CSV 文件来获取基金代码列表。
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在扫描本地目录 {FUND_DATA_DIR} 获取基金代码...")
    fund_codes = []
    
    try:
        for filename in os.listdir(FUND_DATA_DIR):
            if filename.endswith('.csv'):
                # 假设文件名格式为 '基金代码.csv'
                code = filename.replace('.csv', '')
                if re.match(r'^\d{6}$', code): # 简单校验代码格式
                    fund_codes.append(code)
    except FileNotFoundError:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: 未找到本地基金数据目录 {FUND_DATA_DIR}。")
        return pd.DataFrame()

    if not fund_codes:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 警告: 本地目录 {FUND_DATA_DIR} 中未找到有效的基金代码 CSV 文件。")
        return pd.DataFrame()
        
    df = pd.DataFrame(index=fund_codes)
    # 基金名称和3年回报在初始阶段设置为 NaN，将在后续并行处理中尝试填充
    df['name'] = 'N/A' 
    df['rose_3y'] = np.nan 
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功从本地加载 {len(df)} 个基金代码。")

    return df


def calculate_returns(df, end_date_str):
    """从净值数据计算短期回报率 (3年, 1年, 6个月)"""
    # 确保列名正确
    if df.empty or '单位净值' not in df.columns or '净值日期' not in df.columns:
        return {'rose_3y': np.nan, 'rose_1y': np.nan, 'rose_6m': np.nan}
    
    df['净值日期'] = pd.to_datetime(df['净值日期'])
    df.sort_values(by='净值日期', inplace=True)
    
    end_date = pd.to_datetime(end_date_str)
    
    latest_net_value_row = df[df['净值日期'] <= end_date].iloc[-1] if not df[df['净值日期'] <= end_date].empty else None
    
    if latest_net_value_row is None:
        return {'rose_3y': np.nan, 'rose_1y': np.nan, 'rose_6m': np.nan}
        
    latest_date = latest_net_value_row['净值日期']
    latest_net_value = latest_net_value_row['单位净值']
    
    returns = {}

    # Helper function for return calculation
    def get_return(days_ago):
        target_date = latest_date - timedelta(days=days_ago)
        target_row = df[df['净值日期'] <= target_date].iloc[-1] if not df[df['净值日期'] <= target_date].empty else None
        
        if target_row is not None:
            net_value_target_ago = target_row['单位净值']
            return (latest_net_value / net_value_target_ago - 1) * 100
        else:
            return np.nan

    # 3年回报 (rose_3y)
    returns['rose_3y'] = get_return(365 * 3)
    
    # 1年回报 (rose_1y)
    returns['rose_1y'] = get_return(365)

    # 6个月回报 (rose_6m)
    returns['rose_6m'] = get_return(180)
        
    return returns

def download_fund_csv(fund_code, start_date='2020-01-01', end_date=None, force_download=False):
    """
    优先从本地 fund_data/{fund_code}.csv 加载数据，失败或强制下载时回退到网上下载并保存。
    返回一个包含数据帧和状态的字典。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    local_path = os.path.join(FUND_DATA_DIR, f'{fund_code}.csv')
    df_fund = pd.DataFrame()
    loaded_from_local = False
    
    # 1. 尝试从本地加载
    if not force_download and os.path.exists(local_path):
        try:
            df_fund = pd.read_csv(local_path)
            if 'date' in df_fund.columns and 'net_value' in df_fund.columns:
                df_fund.rename(columns={'date': '净值日期', 'net_value': '单位净值'}, inplace=True)
                if not df_fund.empty and '单位净值' in df_fund.columns:
                    loaded_from_local = True
                else:
                    raise ValueError("本地数据为空或格式不正确，尝试下载更新。")
            else:
                raise ValueError("本地CSV文件格式不正确，尝试下载。")
        except Exception:
            df_fund = pd.DataFrame() # 清空数据框以便回退到下载
            
    # 2. 从网上下载 (本地加载失败或强制下载)
    if not loaded_from_local:
        try:
            # 采用东方财富网开放式基金净值走势接口
            df_fund = ak.fund_em_open_fund_info_index(
                fund=fund_code, 
                start_date=start_date.replace('-', ''), 
                end_date=end_date.replace('-', ''), 
                indicator='单位净值走势'
            )
            
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
            df_fund = pd.DataFrame() # 下载失败

    # 3. 数据过滤和返回
    if not df_fund.empty:
        df_fund['净值日期'] = pd.to_datetime(df_fund['净值日期'])
        df_fund = df_fund[(df_fund['净值日期'] >= pd.to_datetime(start_date)) & 
                          (df_fund['净值日期'] <= pd.to_datetime(end_date))]
    
    # 返回数据帧和状态
    return {'df': df_fund, 'loaded_from_local': loaded_from_local}


def get_fund_details(fund_code, cache={}):
    """从网页爬取基金规模和基金经理"""
    if fund_code in cache:
        return cache[fund_code]

    # ... (get_fund_details 函数体保持不变，因为只需要规模和经理，这是非核心数据) ...
    url = f"http://fund.eastmoney.com/{fund_code}.html"
    res = getURL(url)
    details = {'scale': np.nan, 'manager': 'N/A', 'name': fund_code} # 增加 name 字段用于回填

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
                        if '万元' in unit:
                            value /= 10000 
                        details['scale'] = round(value, 2)
        except Exception:
            pass 
        # 2. 基金经理 (manager)
        try:
            manager_tag = soup.find('a', attrs={'href': re.compile(r'/manager/\d+\.html')})
            if manager_tag:
                details['manager'] = manager_tag.text.strip()
        except Exception:
            pass 
        # 3. 基金名称 (name) - 从标题或描述中获取
        try:
            name_tag = soup.find('div', class_='fundDetail-tit')
            if name_tag:
                name_text = name_tag.find('div', class_='dataItem02').text.strip().split('(')[0]
                details['name'] = name_text
        except Exception:
            pass
            
    cache[fund_code] = details
    return details


def analyze_fund(df_fund, start_date, end_date):
    """
    计算基金的夏普比率、波动率和最大回撤。
    输入参数改为 DataFrame，避免重复加载。
    """
    if df_fund.empty:
        return {'error': "没有有效净值数据"}

    # 确保数据在分析期内
    df_fund = df_fund[(df_fund['净值日期'] >= pd.to_datetime(start_date)) & 
                      (df_fund['净值日期'] <= pd.to_datetime(end_date))]

    if df_fund.empty:
        return {'error': "在指定分析期内没有有效净值数据"}

    net_values = df_fund['单位净值']
    # 1. 每日收益率
    daily_returns = net_values.pct_change().dropna()

    if daily_returns.empty:
        return {'error': "每日收益率计算失败"}

    # 2. 波动率 (Volatitlity - 年化标准差)
    std_daily_return = daily_returns.std()
    annualized_volatility = std_daily_return * np.sqrt(ANNUAL_TRADING_DAYS)
    
    # 3. 夏普比率 (Sharpe Ratio)
    mean_daily_return = daily_returns.mean()
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

def process_single_fund(fund_code, start_date, end_date, download_start_date='2020-01-01'):
    """单个基金的并行处理逻辑。"""
    result_data = {
        'code': fund_code,
        'name': 'N/A', # 初始为 N/A
        'rose_3y': np.nan, 
        'rose_1y': np.nan, 
        'rose_6m': np.nan, 
        'scale': np.nan, 
        'manager': 'N/A', 
        'sharpe_ratio': np.nan, 
        'max_drawdown': np.nan,
        'volatility': np.nan,
        'error': None
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] 正在处理基金 {fund_code}...")

    # 1. 下载历史净值 (包含本地优先逻辑)
    # 我们使用更长的下载起始日期来确保能够计算 3 年回报率
    download_result = download_fund_csv(fund_code, start_date=download_start_date, end_date=end_date)
    df_fund = download_result['df']

    if df_fund.empty:
        result_data['error'] = '净值数据获取失败或为空。'
        return result_data
        
    # 2. 计算回报率 (3年, 1年, 6个月)
    returns = calculate_returns(df_fund, end_date)
    result_data.update(returns)
    
    # 3. 获取基金详情（规模、经理、名称）
    details = get_fund_details(fund_code)
    result_data['scale'] = details.get('scale', np.nan)
    result_data['manager'] = details.get('manager', 'N/A')
    result_data['name'] = details.get('name', fund_code) # 填充名称
    
    # 4. 获取并分析风险指标 (使用分析期 start_date 到 end_date)
    risk_metrics = analyze_fund(df_fund, start_date, end_date) 
    if 'error' not in risk_metrics:
        result_data['sharpe_ratio'] = risk_metrics['sharpe_ratio']
        result_data['max_drawdown'] = risk_metrics['max_drawdown']
        result_data['volatility'] = risk_metrics['volatility']
    else:
        result_data['error'] = risk_metrics['error']
        # print(f"[{time.strftime('%H:%M:%S')}] 基金 {fund_code} 风险分析失败: {risk_metrics['error']}")

    return result_data

def get_all_fund_data(fund_type='hh', start_date='2023-10-15', end_date='2025-10-14', limit=100):
    """
    获取基金数据的主函数，使用多线程加速，并基于本地文件列表。
    """
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始获取和分析基金数据...")
    print(f"分析时间范围 (风险指标): {start_date} 至 {end_date}, 无风险利率: {RISK_FREE_RATE*100}%")
    
    # 1. 获取本地基金代码列表 (替换了排行榜爬取)
    ranking_data = get_local_fund_list()
    if ranking_data.empty:
        print("未找到本地基金数据，无法继续。请确保 fund_data 目录下有 CSV 文件。")
        return pd.DataFrame()
        
    # 限制处理数量 (基于本地列表)
    fund_list_to_process = ranking_data.index.tolist()
    if limit > 0:
        fund_list_to_process = fund_list_to_process[:limit]
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 准备处理前 {len(fund_list_to_process)} 只基金...")
    
    final_results = []
    
    # 2. 使用多线程并行处理
    # 设置 download_start_date 确保能覆盖 3 年回报率的计算需求
    download_start_date = (pd.to_datetime(end_date) - timedelta(days=365*3 + 30)).strftime('%Y-%m-%d')
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_fund, code, start_date, end_date, download_start_date): code for code in fund_list_to_process}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
                # print(f"[{time.strftime('%H:%M:%S')}] 完成处理 {i}/{len(fund_list_to_process)} 基金。")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 基金处理过程中发生错误: {e}")
                
    
    if not final_results:
        print("所有基金处理失败或无有效结果。")
        return pd.DataFrame()

    # 3. 数据整合、清洗和排序
    merged_data = pd.DataFrame(final_results).set_index('code')
    
    # 清洗：移除没有成功计算夏普比率或3年回报的行 (防止结果太多无效数据)
    merged_data.replace('N/A', np.nan, inplace=True)
    merged_data.dropna(subset=['rose_3y', 'sharpe_ratio'], how='all', inplace=True)
    
    # 排序：夏普比率降序，3年回报降序
    merged_data.sort_values(by=['sharpe_ratio', 'rose_3y'], ascending=[False, False], inplace=True)
    
    # 格式化输出
    for col in ['rose_3y', 'rose_1y', 'rose_6m', 'max_drawdown', 'volatility']:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce').apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
        
    merged_data['sharpe_ratio'] = pd.to_numeric(merged_data['sharpe_ratio'], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
    merged_data['scale'] = merged_data['scale'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')

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
    
    # 示例: 基金类型参数已失效，但保留以便于将来扩展
    df_result = get_all_fund_data(
        fund_type='local', # 更改为 'local' 以明确数据源
        start_date=start_date, 
        end_date=end_date,
        limit=process_limit
    )
    
    if not df_result.empty:
        output_filename = f"fund_screener_optimized_result_{end_date.replace('-', '')}.xlsx"
        df_result.to_excel(output_filename, index=True, index_label='代码', encoding='utf-8')
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 筛选结果已保存到 {output_filename}")
        print("\n--- 筛选结果预览 (前20条) ---")
        print(df_result.head(20).to_markdown())
    else:
        print("\n未找到符合条件的基金数据。请确保 fund_data 目录下有基金代码.csv 文件。")
