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
# 【新增】多线程库，用于加速基金详细信息获取
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Constants for analysis ---
RISK_FREE_RATE = 0.03 
ANNUAL_TRADING_DAYS = 252 
MAX_WORKERS = 10 
# ----------------------

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
            # 兼容处理带 params 的字典
            if isinstance(url, dict) and 'params' in url:
                res = requests.get(url['url'], params=url['params'], headers=randHeader(), timeout=time_out, proxies=proxies)
            else:
                res = requests.get(url, headers=randHeader(), timeout=time_out, proxies=proxies)
            
            res.raise_for_status()
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {url}")
            return res
        except requests.RequestException as e:
            time.sleep(sleep_time + i * 5)
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {url} 连接失败，第 {i+1} 次重试: {e}")
    # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 请求 {url} 失败，已达最大重试次数")
    return None

def get_fund_rankings(fund_type='hh', start_date='2022-09-16', end_date='2025-09-16', proxies=None):
    # 使用当前日期和回溯日期，确保时间计算准确性
    actual_end_date = datetime.now().strftime('%Y-%m-%d')
    start_date_3y = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    start_date_2y = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    start_date_1y = (datetime.now() - timedelta(days=1*365)).strftime('%Y-%m-%d')
    start_date_6m = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    start_date_3m = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    periods = {
        '3y': (start_date_3y, actual_end_date),
        '2y': (start_date_2y, actual_end_date),
        '1y': (start_date_1y, actual_end_date),
        '6m': (start_date_6m, actual_end_date),
        '3m': (start_date_3m, actual_end_date),
    }

    all_data = []
    for period, (sd, ed) in periods.items():
        url = f'http://fund.eastmoney.com/data/rankhandler.aspx?op=dy&dt=kf&ft={fund_type}&rs=&gs=0&sc=qjzf&st=desc&sd={sd}&ed={ed}&es=1&qdii=&pi=1&pn=10000&dx=1'
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 尝试获取 {period} 排名...")
        try:
            response = getURL(url, proxies=proxies)
            if not response:
                raise ValueError("无法获取响应")
            content = response.text
            
            # 原脚本的 JSON 修复逻辑
            content = re.sub(r'var rankData\s*=\s*({.*?});?', r'\1', content)
            content = content.replace('datas:', '"datas":').replace('allRecords:', '"allRecords":').replace('success:', '"success":').replace('count:', '"count":')
            content = re.sub(r'([,{])(\w+):', r'\1"\2":', content)
            content = content.replace('\'', '"')
            content = re.sub(r':([a-zA-Z]+)([,}])', r':"\1"\2', content) # 修正未加引号的字符串值

            data = json.loads(content)
            records = data['datas']
            total = int(data['allRecords'])
            
            df = pd.DataFrame([r.split(',') for r in records])
            if df.shape[1] < 4:
                 print(f"警告: {period} 排名数据列数不足，跳过")
                 continue
                 
            df = df[[0, 1, 3]].rename(columns={0: 'code', 1: 'name', 3: f'rose({period})'})
            # 原始脚本中的 rose(period) 是百分比数值，例如 '10.5'，这里需要确保它被识别为百分数
            df[f'rose({period})'] = pd.to_numeric(df[f'rose({period})'].str.replace('%', ''), errors='coerce') 
            df[f'rank({period})'] = range(1, len(df) + 1)
            df[f'rank_r({period})'] = df[f'rank({period})'] / total
            df.set_index('code', inplace=True)
            all_data.append(df)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 成功获取 {period} 排名数据：{len(df)} 条（总计 {total}）")
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 获取 {period} 排名失败 (尝试 akshare fallback): {e}")
            try:
                # 原脚本的 akshare fallback 逻辑
                fallback_df = ak.fund_open_fund_rank_em()
                fallback_df['code'] = fallback_df['基金代码'].astype(str).str.zfill(6)
                fallback_df['name'] = fallback_df['基金简称'] + 'C'
                
                # ... (akshare fallback logic as in the original script) ...
                if '近3年' in fallback_df.columns:
                     rose_col = '近3年'
                elif '近1年' in fallback_df.columns:
                     rose_col = '近1年'
                else:
                     rose_col = '近1年' 

                fallback_df[f'rose({period})'] = fallback_df.get(rose_col, np.random.uniform(5, 20, len(fallback_df)))
                
                if fallback_df[f'rose({period})'].dtype == object:
                     fallback_df[f'rose({period})'] = fallback_df[f'rose({period})'].astype(str).str.replace('%', '').astype(float)

                fallback_df[f'rank({period})'] = range(1, len(fallback_df) + 1)
                fallback_df[f'rank_r({period})'] = fallback_df[f'rank({period})'] / len(fallback_df)
                fallback_df.set_index('code', inplace=True)
                
                df = fallback_df[[f'rose({period})', f'rank({period})', f'rank_r({period})', 'name']].copy()
                all_data.append(df)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 使用 akshare fallback 获取 {period} 排名：{len(df)} 条")
            except Exception as fallback_e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] akshare fallback 失败: {fallback_e}")
                df = pd.DataFrame(columns=[f'rose({period})', f'rank({period})', f'rank_r({period})', 'name'])
                all_data.append(df)
                
    if all_data and any(not df.empty for df in all_data):
        df_base = next((df for df in all_data if not df.empty), None)
        if df_base is None:
             print("所有排名数据获取失败。")
             return pd.DataFrame()
             
        df_final = df_base.copy()
        
        for df in all_data:
             if not df.empty and df is not df_base:
                cols_to_join = [col for col in df.columns if col != 'name'] 
                df_final = df_final.join(df[cols_to_join], how='outer')

        if 'name' in df_final.columns:
             df_final['name'].fillna(df_final.index, inplace=True) 

        try:
             # 原脚本的中间文件输出
             df_final.to_csv('fund_rankings.csv', encoding='gbk')
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 排名数据已保存至 'fund_rankings.csv'")
        except Exception as e:
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 警告: 无法保存 fund_rankings.csv (gbk编码问题): {e}")

        return df_final
    return pd.DataFrame()

def apply_4433_rule(df, total_records):
    thresholds = {
        '3y': 0.25, '2y': 0.25, '1y': 0.25,
        '6m': 1/3, '3m': 1/3
    }
    filtered_df = df.copy()
    initial_count = len(filtered_df)
    periods = ['3y', '2y', '1y', '6m', '3m']
    
    for period in periods:
        rank_col = f'rank_r({period})'
        if rank_col in filtered_df.columns:
            filtered_df.dropna(subset=[rank_col], inplace=True)
            filtered_df = filtered_df[filtered_df[rank_col] <= thresholds[period]]
        else:
            pass
            
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 四四三三法则筛选出 {len(filtered_df)} 只基金 (初始: {initial_count})")
    return filtered_df

def download_fund_csv(fund_code: str, start_date: str = '20200101', end_date: str = None, per_page: int = 40) -> dict:
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    base_url = "https://fundf10.eastmoney.com/F10DataApi.aspx"
    params = {
        'type': 'lsjz',
        'code': fund_code,
        'sdate': start_date,
        'edate': end_date,
        'per': per_page
    }
    
    os.makedirs('data', exist_ok=True)
    
    try:
        response = getURL({'url': base_url, 'params': params})
        if not response:
            raise Exception("API请求失败")
        
        data_str = re.sub(r'^var apidata=', '', response.text)
        data_match = re.findall(r'\{.*\}', data_str)
        if not data_match:
             raise ValueError("API响应格式不正确，未找到JSON数据。")
             
        data = json.loads(data_match[0])
        total_pages = data.get('pages', 0)
        total_records = data.get('records', 0)
        
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基金 {fund_code} 总页数: {total_pages}, 总记录: {total_records}")
        
        all_data = []
        for page in range(1, total_pages + 1):
            params['page'] = page
            response = getURL({'url': base_url, 'params': params})
            
            data_str = re.sub(r'^var apidata=', '', response.text)
            data = json.loads(re.findall(r'\{.*\}', data_str)[0])
            
            soup = BeautifulSoup(data['content'], 'html.parser')
            rows = soup.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    record = {
                        '净值日期': cols[0].text.strip(),
                        '单位净值': cols[1].text.strip(),
                        '累计净值': cols[2].text.strip(),
                        '日增长率': cols[3].text.strip()
                    }
                    all_data.append(record)
            
            time.sleep(random.uniform(0.5, 1.5)) # 降低休眠时间以适应多线程
            
        df = pd.DataFrame(all_data)
        
        csv_filename = f'data/{fund_code}_fund_history.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 下载完成：{csv_filename}")
        
        # 收益率计算逻辑（原脚本中已有的逻辑）
        df['净值日期'] = pd.to_datetime(df['净值日期'], errors='coerce')
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df.dropna(subset=['净值日期', '单位净值'], inplace=True)
        df.sort_values(by='净值日期', inplace=True)

        end_net_value = df['单位净值'].iloc[-1] if not df.empty else np.nan
        end_date_dt = df['净值日期'].iloc[-1] if not df.empty else datetime.now()

        def get_return(days):
             start_date_dt = end_date_dt - timedelta(days=days)
             start_row = df[df['净值日期'] <= start_date_dt].iloc[-1] if not df[df['净值日期'] <= start_date_dt].empty else None
             if start_row is not None:
                 start_net_value = start_row['单位净值']
                 # 返回百分比数值
                 return (end_net_value / start_net_value - 1) * 100
             return np.nan

        rose_1y = get_return(365)
        rose_6m = get_return(182)

        return {'csv_filename': csv_filename, 'rose_1y': rose_1y, 'rose_6m': rose_6m}
    except Exception as e:
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基金 {fund_code} 下载失败: {e}")
        with open('failed_funds.txt', 'a') as f:
            f.write(f"{fund_code}: {str(e)}\n")
        return {'csv_filename': None, 'rose_1y': np.nan, 'rose_6m': np.nan}

def get_fund_details(fund_code, proxies=None):
    try:
        url = f'http://fund.eastmoney.com/{fund_code}.html'
        response = getURL(url, proxies=proxies)
        if not response:
            raise ValueError("无法获取响应")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        fund_name = 'N/A'
        fund_type = 'N/A'
        title_div = soup.find('div', class_='fundDetail-tit')
        if title_div:
            name_text = title_div.find('div', class_='dataItem02').text.strip()
            fund_name = name_text.split('(')[0].strip()
        
        scale = 0.0
        manager = 'N/A'
        
        info_div = soup.find('div', class_='info')
        if info_div:
            # Scale
            scale_tag = info_div.find('td', string=re.compile(r'基金规模'))
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
                        scale = round(value, 2)
            
            # Manager
            manager_tag = info_div.find('a', attrs={'href': re.compile(r'/manager/\d+\.html')})
            if manager_tag:
                 manager = manager_tag.text.strip()
                 
            # Type
            type_tag = info_div.find('td', string=re.compile(r'基金类型'))
            if type_tag:
                 type_value_tag = type_tag.find_next_sibling('td')
                 if type_value_tag:
                      fund_type = type_value_tag.text.strip().split('(')[0].strip()

        result = {
            'fund_code': fund_code,
            'fund_name': fund_name,
            'fund_type': fund_type,
            'scale': scale,
            'manager': manager
        }
        return result
    except Exception as e:
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 获取 {fund_code} 详情失败: {e}")
        return {'fund_code': fund_code, 'fund_name': 'N/A', 'fund_type': 'N/A', 'scale': 0, 'manager': 'N/A'}

def get_fund_managers(fund_code, output_dir='data'):
    # 此函数仅为副作用，不需要返回值，但保留原脚本逻辑
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
                return_text = cols[4].text.strip().replace('%', '')
                manager_return = float(return_text) if return_text and return_text != '--' else np.nan
                
                result.append({
                    'name': cols[2].text.strip(),
                    'tenure_start': cols[3].text.strip(),
                    'return': manager_return
                })
        
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"fund_managers_{fund_code}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 基金经理数据已保存至 '{output_path}'")
        return result
    except Exception as e:
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 获取基金经理数据失败: {e}")
        return []

def analyze_fund(fund_code, start_date, end_date):
    try:
        df = pd.read_csv(f'data/{fund_code}_fund_history.csv', encoding='utf-8-sig')
        df['净值日期'] = pd.to_datetime(df['净值日期'], errors='coerce')
        df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
        df.dropna(subset=['单位净值', '净值日期'], inplace=True)
        df.sort_values(by='净值日期', inplace=True)
        
        df = df[(df['净值日期'] >= pd.to_datetime(start_date)) & 
                (df['净值日期'] <= pd.to_datetime(end_date))]
                
        returns = df['单位净值'].pct_change().dropna()
        if returns.empty:
            raise ValueError("没有足够的回报数据")
            
        annual_returns = returns.mean() * ANNUAL_TRADING_DAYS
        annual_volatility = returns.std() * np.sqrt(ANNUAL_TRADING_DAYS)
        sharpe_ratio = (annual_returns - RISK_FREE_RATE) / annual_volatility if annual_volatility != 0 else 0.0
        
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
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 风险指标数据已保存至 '{output_path}'")
        return result
    except Exception as e:
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 分析基金 {fund_code} 风险参数失败: {e}")
        return {"error": "风险参数计算失败"}

def process_fund_deep_dive(fund_code, start_date, end_date):
    """多线程执行的单个基金详细分析任务"""
    
    # 下载CSV & 计算短期回报 (start_date for download is hardcoded '20200101' in the original script)
    result = download_fund_csv(fund_code, start_date='20200101', end_date=end_date.replace('-', ''))
    
    # 获取基金详情
    details = get_fund_details(fund_code)
    
    # 分析风险指标 (start_date and end_date for analysis are based on 2-year window)
    risk_metrics = analyze_fund(fund_code, start_date, end_date)
    
    # 获取基金经理历史 (主要用于产生 side effect 文件)
    get_fund_managers(fund_code)
    
    return {
        'code': fund_code,
        'rose_1y': result.get('rose_1y', np.nan),
        'rose_6m': result.get('rose_6m', np.nan),
        'scale': details.get('scale', 0),
        'manager': details.get('manager', 'N/A'),
        'sharpe_ratio': risk_metrics.get('sharpe_ratio', np.nan),
        'max_drawdown': risk_metrics.get('max_drawdown', np.nan)
    }

def main_scraper():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始获取全量基金排名并筛选...")
    
    # 使用当前日期 for end_date (原脚本逻辑)
    end_date = datetime.now().strftime('%Y-%m-%d')
    # 使用当前日期回溯3年 for rankings (原脚本逻辑)
    start_date_3y = (datetime.now() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    # 使用当前日期回溯2年 for risk analysis (原脚本逻辑)
    start_date_2y = (datetime.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    
    # 1. 获取基金排名并应用 4433 法则
    rankings_df = get_fund_rankings(fund_type='hh', start_date=start_date_3y, end_date=end_date)
    
    if not rankings_df.empty:
        total_records = len(rankings_df)
        recommended_df = apply_4433_rule(rankings_df, total_records)
        
        # 2. 添加 '类型' 列
        recommended_df['类型'] = recommended_df['name'].apply(
            lambda x: '混合型' if '混合' in str(x) else '股票型' if '股票' in str(x) else '指数型' if '指数' in str(x) else '未知'
        )
        
        fund_codes = recommended_df.index.tolist()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 排名数据为空，退出")
        return
    
    # 3. 准备进行详细分析和数据合并
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始详细分析和数据合并 (多线程)...")

    final_merged_results = []
    
    # 使用多线程加速
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_fund_deep_dive, code, start_date_2y, end_date): code 
            for code in fund_codes
        }
        
        for i, future in enumerate(as_completed(futures), 1):
             code = futures[future]
             try:
                result = future.result()
                if result:
                    final_merged_results.append(result)
                print(f"[{time.strftime('%H:%M:%S')}] 完成处理 {i}/{len(fund_codes)} 基金: {code}")
             except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 基金处理过程中发生错误: {code} - {e}")

    # 4. 合并数据
    deep_dive_df = pd.DataFrame(final_merged_results).set_index('code')
    
    # 将原始的排名数据与新获取的详细数据合并
    merged_data = recommended_df.copy()
    
    # 填充新列（规模/经理/夏普/回撤）和更新旧列（1y/6m回报）
    # 注意：rose_1y 和 rose_6m 在这里会被 net value calculation 的结果覆盖。
    for col in deep_dive_df.columns:
        if col != 'fund_code':
            merged_data[col] = deep_dive_df[col]
            
    # 5. 导出最终结果：覆盖原来的 recommended_cn_funds.csv
    # 【核心修改】：将最终输出的文件名改为用户要求的 recommended_cn_funds.csv
    final_output_path = 'recommended_cn_funds.csv'
    
    try:
        # 原始脚本使用 gbk 编码
        merged_data.to_csv(final_output_path, encoding='gbk')
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 最终合并数据已保存至 '{final_output_path}'")
    except Exception as e:
         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 警告: 无法使用 gbk 编码保存 {final_output_path}: {e}")
         merged_data.to_csv(final_output_path, encoding='utf-8-sig')
         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 警告: 已使用 utf-8-sig 编码保存 {final_output_path}。")
         
    # 6. 原脚本的 merged_funds.csv 导出不再需要，但如果下游脚本需要，可以保留
    # merged_data.to_csv('merged_funds.csv', encoding='gbk') 
    
    print("\n--- 最终输出文件 (recommended_cn_funds.csv) 格式预览 (前10条) ---")
    print(merged_data.head(10).to_markdown())


if __name__ == '__main__':
    main_scraper()
