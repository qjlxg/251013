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
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 全局配置 ---
MAX_THREADS = 5  # 并行下载的最大线程数
FAILED_FUNDS_FILE = 'failed_funds.txt'
MAX_RETRIES = 3  # 失败基金的最大重试次数
LOCAL_DATA_DIR = 'fund_data' # 本地历史数据目录 (优先查找)

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

# 核心修复: 添加 **kwargs 以支持 requests.get 的 params 参数
def getURL(url, tries_num=5, sleep_time=1, time_out=15, proxies=None, **kwargs):
    for i in range(tries_num):
        try:
            time.sleep(random.uniform(0.1, sleep_time)) 
            res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies, **kwargs)
            res.raise_for_status()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 1)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_rankings(fund_type='hh', start_date='2022-09-16', end_date='2025-09-16', proxies=None):
    periods = {
        '3y': (start_date, end_date),
        '2y': (f"{int(end_date[:4])-2}{end_date[4:]}", end_date),
        '1y': (f"{int(end_date[:4])-1}{end_date[4:]}", end_date),
        '6m': ((datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d'), end_date),
        '3m': ((datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d'), end_date)
    }
    all_data = []
    
    akshare_period_map = {
        '3y': '近3年', '2y': '近2年', '1y': '近1年',
        '6m': '近6月', '3m': '近3月'
    }

    for period, (sd, ed) in periods.items():
        url = f'http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={sd}&ed={ed}&es=1&qdii=&pi=1&pn=10000&dx=1'
        try:
            response = getURL(url, proxies=proxies)
            if not response:
                raise ValueError("无法获取响应")
            content = response.text
            content = re.sub(r'var rankData\s*=\s*({.*?});?', r'\1', content)
            content = content.replace('datas:', '"datas":').replace('allRecords:', '"allRecords":').replace('success:', '"success":').replace('count:', '"count":')
            content = re.sub(r'([,{])(\w+):', r'\1"\2":', content)
            content = content.replace('\'', '"')
            data = json.loads(content)
            records = data['datas']
            total = int(data['allRecords'])
            
            if not records:
                 raise ValueError("东方财富网API返回空数据")

            df = pd.DataFrame([r.split(',') for r in records])
            
            df = df[[0, 1, 3]].rename(columns={0: 'code', 1: 'name', 3: f'rose({period})'})
            
            df[f'rose({period})'] = pd.to_numeric(df[f'rose({period})'].replace('--', np.nan).str.replace('%', ''), errors='coerce') / 100
            
            df.dropna(subset=['code'], inplace=True) 
            df.set_index('code', inplace=True)
            
            df.dropna(subset=[f'rose({period})'], inplace=True)
            df[f'rank({period})'] = df[f'rose({period})'].rank(ascending=False, method='min')
            df[f'rank_r({period})'] = df[f'rank({period})'] / total

            all_data.append(df)
            print(f"获取 {period} 排名数据：{len(df)} 条（总计 {total}）")
            
        except Exception as e:
            print(f"获取 {period} 排名失败 ({e})，尝试使用 akshare fallback...")
            try:
                fallback_df = ak.fund_open_fund_rank_em()
                fallback_df['code'] = fallback_df['基金代码'].astype(str).str.zfill(6)
                
                ak_col = akshare_period_map.get(period)
                if ak_col not in fallback_df.columns:
                    print(f"akshare 中缺少 {period} 对应的字段 ({ak_col})，跳过该周期。")
                    continue

                fallback_df['name'] = fallback_df['基金简称']
                
                fallback_df[f'rose({period})'] = pd.to_numeric(fallback_df[ak_col], errors='coerce') / 100
                
                fallback_df.dropna(subset=['code', f'rose({period})'], inplace=True)
                total_ak = len(fallback_df)
                fallback_df[f'rank({period})'] = fallback_df[f'rose({period})'].rank(ascending=False, method='min')
                fallback_df[f'rank_r({period})'] = fallback_df[f'rank({period})'] / total_ak
                fallback_df.set_index('code', inplace=True)
                
                df = fallback_df[[f'rose({period})', f'rank({period})', f'rank_r({period})', 'name']]
                all_data.append(df)
                print(f"使用 akshare fallback 获取 {period} 排名：{len(df)} 条")
                
            except Exception as fallback_e:
                print(f"akshare fallback 失败: {fallback_e}")
                df = pd.DataFrame(columns=[f'rose({period})', f'rank({period})', f'rank_r({period})', 'name'])
                all_data.append(df)
                
    if all_data and any(not df.empty for df in all_data):
        df_final = next(df.copy() for df in all_data if not df.empty)
        
        for df in all_data:
            if not df.empty and df is not df_final:
                cols_to_join = [col for col in df.columns if col != 'name'] 
                df_final = df_final.join(df[cols_to_join], how='outer')

        rank_cols = [f'rank_r({p})' for p in periods.keys()]
        df_final.dropna(subset=rank_cols, thresh=len(rank_cols) - 2, inplace=True)
        
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
    initial_count = len(filtered_df)
    
    for period in thresholds:
        rank_col = f'rank_r({period})'
        if rank_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[rank_col] <= thresholds[period]]
            print(f" - {period} 筛选后剩余 {len(filtered_df)} 只")

    print(f"四四三三法则筛选出 {len(filtered_df)} 只基金 (初始: {initial_count})")
    return filtered_df

def download_fund_csv(fund_code: str, start_date: str = '20200101', end_date: str = None, per_page: int = 40) -> dict:
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
        
    local_csv_path = os.path.join(LOCAL_DATA_DIR, f'{fund_code}.csv')
    temp_csv_path = f'data/{fund_code}_fund_history.csv' 
    df_final = None
    
    # 1. 尝试从本地目录读取 (fund_data)
    is_local_data_recent = False
    
    if os.path.exists(local_csv_path):
        try:
            df_local = pd.read_csv(local_csv_path, encoding='utf-8-sig')
            
            # --- 列名映射：尝试匹配 'date/net_value' 或使用前两列 ---
            if 'date' in df_local.columns and 'net_value' in df_local.columns:
                df_local.rename(columns={'date': '净值日期', 'net_value': '单位净值'}, inplace=True)
            elif '净值日期' not in df_local.columns or '单位净值' not in df_local.columns:
                 if len(df_local.columns) >= 2:
                     # 假设前两列是日期和净值
                     df_local.rename(columns={df_local.columns[0]: '净值日期', df_local.columns[1]: '单位净值'}, inplace=True)
                 else:
                     raise ValueError("本地文件数据列数不足")

            # 确保数据类型正确并清理
            df_local['净值日期'] = pd.to_datetime(df_local['净值日期'], errors='coerce')
            df_local['单位净值'] = pd.to_numeric(df_local['单位净值'], errors='coerce')
            df_local.dropna(subset=['净值日期', '单位净值'], inplace=True)
            df_local.sort_values(by='净值日期', inplace=True)

            if not df_local.empty:
                max_date = df_local['净值日期'].max()
                # 完整性/时效性检查：如果数据是最近两天内的，认为完整
                if (datetime.now() - max_date) < timedelta(days=2):
                    df_final = df_local
                    is_local_data_recent = True
                    print(f"使用本地完整数据：{local_csv_path} (最新日期: {max_date.date()})")
                else:
                    print(f"本地数据 {local_csv_path} 已过期 (最新日期: {max_date.date()})，将尝试网络下载更新。")
            
        except Exception as e:
            print(f"读取或解析本地文件 {local_csv_path} 失败: {e}，将尝试网络下载。")
            
    # 2. 如果本地数据不完整或过期，则进行网络下载
    if not is_local_data_recent:
        # 完整的网络下载逻辑
        base_url = "https://fundf10.eastmoney.com/F10DataApi.aspx"
        params = {'type': 'lsjz', 'code': fund_code, 'sdate': start_date, 'edate': end_date, 'per': per_page}
        
        try:
            response = getURL(base_url, params=params)
            if not response:
                raise Exception("API请求失败")
            
            data_str = re.sub(r'^var apidata=', '', response.text)
            data_str = re.findall(r'\{.*\}', data_str)[0]
            data = json.loads(data_str)
            total_pages = data['pages']
            all_data = []
            
            for page in range(1, total_pages + 1):
                params['page'] = page
                response = getURL(base_url, params=params, sleep_time=1)
                if not response: break
                
                data_str = re.sub(r'^var apidata=', '', response.text)
                data_str = re.findall(r'\{.*\}', data_str)[0]
                data = json.loads(data_str)
                soup = BeautifulSoup(data['content'], 'html.parser')
                rows = soup.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        record = {'净值日期': cols[0].text.strip(), '单位净值': cols[1].text.strip(), 
                                  '累计净值': cols[2].text.strip(), '日增长率': cols[3].text.strip()}
                        all_data.append(record)
            
            if not all_data:
                raise Exception("未下载到任何净值数据")
                
            df_download = pd.DataFrame(all_data)
            df_download['净值日期'] = pd.to_datetime(df_download['净值日期'])
            df_download['单位净值'] = pd.to_numeric(df_download['单位净值'], errors='coerce')
            df_download.sort_values(by='净值日期', inplace=True)
            
            # --- 保存到本地目录 (fund_data)：如果下载成功，覆盖本地数据 ---
            os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
            df_download.to_csv(local_csv_path, index=False, encoding='utf-8-sig')
            print(f"下载完成并保存到本地目录：{local_csv_path}")
            df_final = df_download
            
        except Exception as e:
            # 如果网络下载失败，并且我们有旧的本地数据，则退回到旧数据
            if df_final is None and os.path.exists(local_csv_path):
                 print("网络下载失败，退回使用旧的本地数据 (可能不完整)。")
                 # 重新加载旧数据（这次不检查时效性）
                 try:
                     df_local = pd.read_csv(local_csv_path, encoding='utf-8-sig')
                     if 'date' in df_local.columns and 'net_value' in df_local.columns:
                         df_local.rename(columns={'date': '净值日期', 'net_value': '单位净值'}, inplace=True)
                     elif '净值日期' not in df_local.columns or '单位净值' not in df_local.columns:
                          if len(df_local.columns) >= 2:
                             df_local.rename(columns={df_local.columns[0]: '净值日期', df_local.columns[1]: '单位净值'}, inplace=True)
                          else:
                             raise ValueError("列数不足，无法恢复")

                     df_local['净值日期'] = pd.to_datetime(df_local['净值日期'], errors='coerce')
                     df_local['单位净值'] = pd.to_numeric(df_local['单位净值'], errors='coerce')
                     df_local.dropna(subset=['净值日期', '单位净值'], inplace=True)
                     df_local.sort_values(by='净值日期', inplace=True)
                     df_final = df_local
                 except:
                     pass # 无法挽救
            
            if df_final is None or df_final.empty:
                 raise Exception(f"网络下载失败且无有效本地数据: {e}")

    # 3. 最终处理和计算回报
    if df_final is None or df_final.empty:
        raise Exception("无法获取任何历史净值数据")
    
    # 总是保存到 data/ 目录供 analyze_fund 使用
    os.makedirs('data', exist_ok=True)
    
    # 确保最终 df 包含 analyze_fund 需要的所有列
    for col in ['累计净值', '日增长率']:
        if col not in df_final.columns:
            df_final[col] = np.nan
            
    df_final.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
    
    # 计算回报
    one_year_ago = datetime.now() - timedelta(days=365)
    six_month_ago = datetime.now() - timedelta(days=182)
    recent_data = df_final[df_final['净值日期'] >= one_year_ago]
    six_month_data = df_final[df_final['净值日期'] >= six_month_ago]
    
    # 确保有足够的样本点来计算回报
    rose_1y = (recent_data['单位净值'].iloc[-1] / recent_data['单位净值'].iloc[0] - 1) * 100 if len(recent_data) > 1 else np.nan
    rose_6m = (six_month_data['单位净值'].iloc[-1] / six_month_data['单位净值'].iloc[0] - 1) * 100 if len(six_month_data) > 1 else np.nan
    
    return {'csv_filename': temp_csv_path, 'rose_1y': rose_1y, 'rose_6m': rose_6m}


def get_fund_details(fund_code, proxies=None):
    try:
        url = f'http://fund.eastmoney.com/f10/{fund_code}.html'
        response = getURL(url, proxies=proxies)
        if not response:
            raise ValueError("无法获取响应")
        tables = pd.read_html(io.StringIO(response.text))
        if len(tables) < 2:
            raise ValueError("表格数量不足")
        df_info = tables[0].T.set_index(0).to_dict()
        df = df_info[list(df_info.keys())[0]]
        fund_name = df.get('基金全称', 'N/A')
        fund_type = df.get('基金类型', 'N/A')
        manager = df.get('基金经理', 'N/A')
        scale_str = df.get('基金规模', '0')
        scale = 0
        if '亿元' in scale_str:
             scale = float(scale_str.replace('亿元', ''))
        elif '万元' in scale_str:
             scale = float(scale_str.replace('万元', '')) / 10000
        result = {
            'fund_code': fund_code,
            'fund_name': fund_name,
            'fund_type': fund_type,
            'scale': scale,
            'manager': manager
        }
        return result
    except Exception as e:
        return {'fund_code': fund_code, 'fund_name': 'N/A', 'fund_type': 'N/A', 'scale': 0, 'manager': 'N/A'}

def get_fund_managers(fund_code, output_dir='data'):
    try:
        url = f'http://fundf10.eastmoney.com/jjjl_{fund_code}.html'
        response = getURL(url, sleep_time=1)
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
                return_text = cols[4].text.strip()
                fund_return = float(return_text.replace('%', '')) if '%' in return_text and return_text != '--' else np.nan
                result.append({
                    'name': cols[2].text.strip(),
                    'tenure_start': cols[3].text.strip(),
                    'return': fund_return
                })
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"fund_managers_{fund_code}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result
    except Exception as e:
        return []

def analyze_fund(fund_code, start_date, end_date):
    try:
        csv_path = f'data/{fund_code}_fund_history.csv'
        if not os.path.exists(csv_path):
             return {"error": "历史数据文件不存在"}
             
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        
        df.sort_values(by='净值日期', inplace=True)
        df.dropna(subset=['单位净值'], inplace=True)
        
        returns = df['单位净值'].pct_change().dropna()
        if returns.empty or len(returns) < 50:
            raise ValueError(f"没有足够的回报数据（样本点 < 50）")
            
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
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result
    except Exception as e:
        print(f"分析基金 {fund_code} 风险参数失败: {e}")
        return {"error": "风险参数计算失败"}

def process_single_fund(fund_code, start_date, end_date, attempt=1):
    """处理单个基金的下载、详情获取和分析的全部流程"""
    print(f"[Attempt {attempt}] 开始处理基金 {fund_code}...")
    
    fund_data = {
        'code': fund_code,
        'rose_1y': np.nan, 'rose_6m': np.nan,
        'scale': np.nan, 'manager': 'N/A',
        'sharpe_ratio': np.nan, 'max_drawdown': np.nan
    }
    
    try:
        # 1. 下载历史净值并计算短期回报 (优先使用本地数据，过期则下载并更新 fund_data/)
        result = download_fund_csv(fund_code, start_date='20200101', end_date=end_date)
        if result['csv_filename']:
            fund_data['rose_1y'] = result['rose_1y']
            fund_data['rose_6m'] = result['rose_6m']
            
        # 2. 获取基金详情
        details = get_fund_details(fund_code)
        fund_data['scale'] = details.get('scale', 0)
        fund_data['manager'] = details.get('manager', 'N/A')
        
        # 3. 获取并分析风险指标 (依赖于第1步成功)
        risk_metrics = analyze_fund(fund_code, start_date, end_date)
        if 'error' not in risk_metrics:
            fund_data['sharpe_ratio'] = risk_metrics.get('sharpe_ratio', np.nan)
            fund_data['max_drawdown'] = risk_metrics.get('max_drawdown', np.nan)
            
        # 4. 获取经理数据
        get_fund_managers(fund_code)
        
        print(f"基金 {fund_code} 处理完成。")
        return fund_data
        
    except Exception as e:
        print(f"基金 {fund_code} 整体处理失败: {e}")
        return None

def main_scraper():
    print("开始获取全量基金排名并筛选...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    rankings_df = get_fund_rankings(fund_type='hh', start_date=start_date, end_date=end_date)
    
    if not rankings_df.empty:
        total_records = len(rankings_df)
        recommended_df = apply_4433_rule(rankings_df, total_records)
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
    
    for col in ['rose_1y', 'rose_6m', 'scale', 'manager', 'sharpe_ratio', 'max_drawdown']:
        if col not in merged_data.columns:
            merged_data[col] = np.nan if col not in ['manager'] else 'N/A'
    
    print(f"\n开始使用 {MAX_THREADS} 线程并行处理 {len(fund_codes)} 只基金...")
    
    if os.path.exists(FAILED_FUNDS_FILE):
        os.remove(FAILED_FUNDS_FILE)
    
    all_results = []
    
    # 第一次处理
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_code = {
            executor.submit(process_single_fund, code, start_date, end_date, 1): code 
            for code in fund_codes
        }
        for future in as_completed(future_to_code):
            result = future.result()
            if result:
                all_results.append(result)

    # --- 批量重试机制 ---
    
    for attempt in range(2, MAX_RETRIES + 1):
        if not os.path.exists(FAILED_FUNDS_FILE) or os.path.getsize(FAILED_FUNDS_FILE) == 0:
            break
            
        with open(FAILED_FUNDS_FILE, 'r') as f:
            failed_codes = [line.split(':')[0].strip() for line in f.readlines()]
            failed_codes = list(set(failed_codes))
            
        print(f"\n--- 第 {attempt} 轮重试：处理 {len(failed_codes)} 只失败基金 ---")
        
        os.remove(FAILED_FUNDS_FILE) 
        
        retry_results = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_code = {
                executor.submit(process_single_fund, code, start_date, end_date, attempt): code 
                for code in failed_codes
            }
            for future in as_completed(future_to_code):
                result = future.result()
                if result:
                    retry_results.append(result)
        
        all_results.extend(retry_results)

    # --- 结果合并与输出 ---
    
    if all_results:
        results_df = pd.DataFrame(all_results).set_index('code')
        merged_data.update(results_df) 

    merged_path = 'merged_funds.csv'
    merged_data.to_csv(merged_path, encoding='gbk')
    print(f"\n--- 最终结果 ---")
    print(f"合并数据（含1年/6月回报、规模等）保存至 '{merged_path}'")
    if os.path.exists(FAILED_FUNDS_FILE) and os.path.getsize(FAILED_FUNDS_FILE) > 0:
         print(f"注意: 仍有基金处理失败，请检查 '{FAILED_FUNDS_FILE}' 文件。")
    elif os.path.exists(FAILED_FUNDS_FILE):
         os.remove(FAILED_FUNDS_FILE)

if __name__ == '__main__':
    main_scraper()
