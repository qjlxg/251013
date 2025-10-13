import pandas as pd
import requests
import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import json

# 定义文件路径和目录
INPUT_FILE = 'recommended_cn_funds.csv'
OUTPUT_DIR = 'fund_data'
# 使用天天基金 API 接口，每页 20 条
BASE_URL = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"

# 设置请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/533.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30 
REQUEST_DELAY = 0.5  # 初始延迟，动态调整
MAX_CONCURRENT = 5  # 最大并发基金数量

def get_all_fund_codes(file_path):
    """【加速读取】从 CSV 文件中仅读取 'code' 列，并尝试不同编码"""
    print(f"尝试读取基金代码文件 (仅读取 'code' 列): {file_path}")
    encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, dtype={'code': str}, usecols=['code'])
            print(f"  -> 成功使用 {encoding} 编码读取文件。")
            break
        except UnicodeDecodeError:
            continue
        except ValueError:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
            except Exception:
                continue
            
    if df is None:
        return []

    if 'code' in df.columns:
        codes = df['code'].dropna().unique().tolist()
        return [code.strip() for code in codes if code.strip()]
    else:
        if len(df.columns) > 0:
            first_col_name = df.columns[0]
            codes = df[first_col_name].dropna().astype(str).unique().tolist()
            return [code.strip() for code in codes if code.strip()]
        else:
            return []

def load_cache(fund_code):
    """加载缓存，获取已爬取的页面"""
    cache_file = os.path.join(OUTPUT_DIR, f"{fund_code}_cache.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f).get('last_page', 0)
        except Exception:
            return 0
    return 0

def save_cache(fund_code, last_page):
    """保存缓存，记录已爬取的页面"""
    cache_file = os.path.join(OUTPUT_DIR, f"{fund_code}_cache.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump({'last_page': last_page}, f)
    except Exception:
        pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_message(match='Frequency Capped')
)
async def fetch_page(session, url):
    """异步请求单页数据，带重试机制"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 514:
            raise aiohttp.ClientError("Frequency Capped")
        response.raise_for_status()
        return await response.text()

async def fetch_net_values(fund_code, session, semaphore):
    """使用天天基金 API 获取指定基金代码的【所有】历史净值数据 (分页，异步)"""
    print(f"-> 正在使用天天基金 API 获取基金代码 {fund_code} 的所有历史净值...")
    
    async with semaphore:  # 控制并发
        all_records = []
        page_index = load_cache(fund_code) + 1  # 从缓存的下一页开始
        page_size = 20
        total_pages = 1
        first_run = True
        dynamic_delay = REQUEST_DELAY

        while page_index <= total_pages:
            url = BASE_URL.format(fund_code=fund_code, page_index=page_index, page_size=page_size)
            
            try:
                if page_index > 1:
                    await asyncio.sleep(dynamic_delay) 
                
                text = await fetch_page(session, url)
                soup = BeautifulSoup(text, 'lxml')
                
                if first_run:
                    total_pages_match = re.search(r'pages:(\d+)', text)
                    total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
                    records_match = re.search(r'records:(\d+)', text)
                    total_records = int(records_match.group(1)) if records_match else '未知'
                    print(f"   基金总页数: {total_pages}，总记录数: {total_records}")
                    first_run = False
                
                table = soup.find('table') 

                if not table:
                    break

                rows = table.find_all('tr')[1:] 

                if not rows:
                    break

                new_records_count = 0
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        date_str = cols[0].text.strip()
                        net_value_str = cols[1].text.strip() 
                        
                        if date_str and net_value_str and net_value_str != '-':
                            all_records.append({'date': date_str, 'net_value': net_value_str})
                            new_records_count += 1

                print(f"   已获取第 {page_index}/{total_pages} 页，新增 {new_records_count} 条记录，总数: {len(all_records)}")
                
                save_cache(fund_code, page_index)  # 保存缓存
                page_index += 1
                dynamic_delay = max(REQUEST_DELAY, dynamic_delay * 0.9)  # 动态减少延迟

            except aiohttp.ClientError as e:
                if "Frequency Capped" in str(e):
                    dynamic_delay = min(dynamic_delay * 2, 5.0)  # 频率限制时增加延迟
                    print(f"   频率限制，调整延迟为 {dynamic_delay} 秒，重试第 {page_index} 页")
                    continue
                print(f"   请求 API 时发生网络错误 (超时/连接) 在第 {page_index} 页: {e}")
                break
            except Exception as e:
                print(f"   处理数据时发生意外错误在第 {page_index} 页: {e}")
                break
            
        return fund_code, all_records

def save_to_csv(fund_code, data):
    """将历史净值数据以增量更新方式保存为 CSV 文件，格式为 date,net_value"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    new_df = pd.DataFrame(data)

    if new_df.empty:
        return

    try:
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date', 'net_value'], inplace=True)
    except Exception:
        return
    
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, parse_dates=['date'], dtype={'net_value': float}, encoding='utf-8')
            combined_df = pd.concat([new_df, existing_df])
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df
        
    final_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    final_df = final_df.sort_values(by='date', ascending=False)
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    try:
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"   成功增量保存数据到 {output_path}，总记录数: {len(final_df)}。")
    except Exception:
        pass

async def fetch_all_funds(fund_codes):
    """异步获取所有基金数据，控制并发"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with ClientSession() as session:
        tasks = [fetch_net_values(fund_code, session, semaphore) for fund_code in fund_codes]
        return await asyncio.gather(*tasks, return_exceptions=True)

def main():
    """主函数"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    fund_codes = get_all_fund_codes(INPUT_FILE)
    if not fund_codes:
        print("没有可处理的基金代码，脚本结束。")
        return

    if len(fund_codes) > 100:
        print(f"警告：检测到 {len(fund_codes)} 个基金代码，为避免 GitHub Actions 超时，本次运行将限制为前 100 个基金。")
        fund_codes = fund_codes

    print(f"找到 {len(fund_codes)} 个基金代码，开始获取历史净值...")
    
    results = asyncio.run(fetch_all_funds(fund_codes))
    
    for fund_code, net_values in results:
        if isinstance(net_values, list):
            print("-" * 30)
            save_to_csv(fund_code, net_values)
        
    print("\n本次基金的历史净值数据获取和保存完成。")

if __name__ == "__main__":
    main()
