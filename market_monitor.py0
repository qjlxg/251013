import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime, timedelta, time
import random
from io import StringIO
import requests
import tenacity
import concurrent.futures
import time as time_module

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 修改：启用调试日志以捕获详细信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义本地数据存储目录
DATA_DIR = 'fund_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class MarketMonitor:
    def __init__(self, report_file='recommended_cn_funds.csv', output_file='market_monitor_report.md'):
        self.report_file = report_file
        self.output_file = output_file
        self.fund_codes = []
        self.fund_data = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Referer': 'http://fundf10.eastmoney.com/',  # 修改：添加 Referer 防反爬
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9'
        }
        # 新增：加载沪深300指数数据
        self.index_data = self._load_index_data()

    # 新增：加载沪深300指数数据
    def _load_index_data(self):
        """加载沪深300指数数据 (000300.csv)"""
        index_file = 'index_data/000300.csv'
        logger.info("正在加载沪深300指数数据 %s...", index_file)
        if not os.path.exists(index_file):
            logger.error("沪深300指数数据文件 %s 不存在。", index_file)
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(index_file, parse_dates=['date'], encoding='utf-8')
            if not df.empty and 'date' in df.columns and 'net_value' in df.columns:
                logger.info("成功加载沪深300指数数据，记录数: %d", len(df))
                return df
            else:
                logger.warning("沪深300指数数据文件 %s 格式不正确或数据为空。", index_file)
                return pd.DataFrame()
        except Exception as e:
            logger.error("读取沪深300指数数据失败: %s", e)
            return pd.DataFrame()

    def _get_expected_latest_date(self):
        """根据当前时间确定期望的最新数据日期 (使用北京时间)"""
        now_utc = datetime.utcnow()
        EIGHT_HOURS = timedelta(hours=8)
        now_cst = now_utc + EIGHT_HOURS
        
        update_time = time(21, 0)
        
        if now_cst.time() < update_time:
            expected_date = now_cst.date() - timedelta(days=1)
        else:
            expected_date = now_cst.date()
            
        logger.info("当前北京时间: %s, 期望最新数据日期: %s", now_cst.strftime('%Y-%m-%d %H:%M:%S'), expected_date)
        return expected_date

    def _parse_report(self):
        """从 recommended_cn_funds.csv 提取推荐基金代码 (已修复编码问题)"""
        logger.info("正在解析 %s 获取推荐基金代码...", self.report_file)
        if not os.path.exists(self.report_file):
            logger.error("文件 %s 不存在。", self.report_file)
            raise FileNotFoundError(f"推荐基金文件 {self.report_file} 不存在")
        
        try:
            df = pd.read_csv(self.report_file, dtype={'code': str}, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(self.report_file, dtype={'code': str}, encoding='gbk')
            except Exception as e:
                logger.error("读取文件失败，请检查文件编码或格式: %s", e)
                raise

        if 'code' not in df.columns:
            logger.error("文件 %s 中缺少 'code' 列，无法提取基金代码。", self.report_file)
            raise ValueError(f"文件 {self.report_file} 中缺少 'code' 列")

        extracted_codes = set(df['code'].dropna().unique())
        sorted_codes = sorted(list(extracted_codes))
        self.fund_codes = sorted_codes[:10]  # 修改：限制基金数量为10个，便于调试
        
        if not self.fund_codes:
            logger.warning("未提取到任何有效基金代码，请检查 %s 文件中 'code' 列的数据。", self.report_file)
        else:
            logger.info("提取到 %d 个基金代码: %s", len(self.fund_codes), self.fund_codes)

    def _read_local_data(self, fund_code):
        """读取本地文件，如果存在则返回DataFrame"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                if not df.empty and 'date' in df.columns and 'net_value' in df.columns:
                    logger.info("成功读取本地数据 %s，记录数: %d", file_path, len(df))
                    return df
            except Exception as e:
                logger.warning("读取本地文件 %s 失败: %s", file_path, e)
        return pd.DataFrame()

    def _save_to_local_file(self, fund_code, df):
        """将DataFrame保存到本地文件，覆盖旧文件"""
        file_path = os.path.join(DATA_DIR, f"{fund_code}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            df.to_csv(file_path, index=False)
            logger.info("基金 %s 数据已成功保存到本地文件，记录数: %d", fund_code, len(df))
        except Exception as e:
            logger.error("保存基金 %s 数据到 %s 失败: %s", fund_code, file_path, e)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_fixed(10),
        retry=tenacity.retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
        before_sleep=lambda retry_state: logger.info(f"重试基金 {retry_state.args[0]}，第 {retry_state.attempt_number} 次")
    )
    def _fetch_fund_data(self, fund_code):
        """从网络获取基金数据，数据不足时增量下载更多页面"""
        logger.info("开始获取基金 %s 数据...", fund_code)
        local_df = self._read_local_data(fund_code)
        latest_local_date = local_df['date'].max().date() if not local_df.empty else None
        expected_latest_date = self._get_expected_latest_date()
        min_data_points = 50  # 修改：定义最小数据要求
        
        all_new_data = []
        page_index = 1
        max_pages_to_check = 10  # 修改：最大页面数
        
        while page_index <= max_pages_to_check:
            url = f"http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"
            logger.debug("请求URL: %s", url)
            
            try:
                response = requests.get(url, headers=self.headers, timeout=60)
                logger.debug("基金 %s 页面 %d HTTP状态码: %d", fund_code, page_index, response.status_code)
                response.raise_for_status()
                
                # 修改：记录响应内容摘要
                response_text = response.text[:200] if len(response.text) > 200 else response.text
                logger.debug("基金 %s 页面 %d 响应内容摘要: %s", fund_code, page_index, response_text)
                
                content_match = re.search(r'content:"(.*?)"', response.text, re.S)
                if not content_match:
                    logger.warning("基金 %s 页面 %d 无有效内容，可能基金代码无效或网站限制", fund_code, page_index)
                    break

                raw_content_html = content_match.group(1).replace('\\"', '"')
                try:
                    tables = pd.read_html(StringIO(raw_content_html))
                    logger.debug("基金 %s 页面 %d 解析到 %d 张表格", fund_code, page_index, len(tables))
                except ValueError as e:
                    logger.warning("基金 %s 页面 %d 数据解析失败: %s", fund_code, page_index, e)
                    break
                if not tables or len(tables) == 0:
                    logger.warning("基金 %s 页面 %d 无表格数据，可能数据不可用", fund_code, page_index)
                    break
                
                df = tables[0]
                if len(df.columns) < 7:
                    logger.warning("基金 %s 页面 %d 表格列数不足: %d", fund_code, page_index, len(df.columns))
                    break
                
                df.columns = ['date', 'net_value', 'cumulative_net_value', 'daily_growth_rate', 'purchase_status', 'redemption_status', 'dividend']
                df = df[['date', 'net_value']].copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['net_value'] = pd.to_numeric(df['net_value'], errors='coerce')
                df = df.dropna(subset=['date', 'net_value'])
                
                if df.empty:
                    logger.warning("基金 %s 页面 %d 数据为空或格式无效", fund_code, page_index)
                    break
                
                new_df = df[df['date'].dt.date > latest_local_date] if latest_local_date else df
                if not new_df.empty:
                    all_new_data.append(new_df)
                    logger.info("基金 %s 页面 %d 获取到 %d 条新数据", fund_code, page_index, len(new_df))
                
                # 修改：检查总数据量是否足够
                total_data = pd.concat([local_df, *all_new_data]).drop_duplicates(subset=['date'], keep='last')
                if len(total_data) >= min_data_points:
                    logger.info("基金 %s 数据已满足要求，记录数: %d，停止获取，页面: %d", fund_code, len(total_data), page_index)
                    break
                if len(df) < 20:
                    logger.info("基金 %s 页面 %d 数据不足20条，无更多数据，停止获取", fund_code, page_index)
                    break
                if not new_df.empty and new_df['date'].max().date() >= expected_latest_date and len(total_data) >= 20:
                    logger.info("基金 %s 数据已包含最新日期 %s，记录数: %d，停止获取，页面: %d", 
                                fund_code, expected_latest_date, len(total_data), page_index)
                    break
                
                page_index += 1
                time_module.sleep(random.uniform(2, 5))  # 修改：延长随机延迟以避免反爬
                
            except requests.exceptions.RequestException as e:
                logger.error("基金 %s 页面 %d 请求失败: %s", fund_code, page_index, e)
                raise
            except Exception as e:
                logger.error("基金 %s 页面 %d 数据处理异常: %s", fund_code, page_index, e)
                raise

        if all_new_data:
            new_combined_df = pd.concat(all_new_data, ignore_index=True)
            df_final = pd.concat([local_df, new_combined_df]).drop_duplicates(subset=['date'], keep='last').sort_values(by='date', ascending=True)
            self._save_to_local_file(fund_code, df_final)
            logger.info("基金 %s 数据更新完成，记录数: %d", fund_code, len(df_final))
            if len(df_final) < min_data_points:
                logger.warning("基金 %s 更新后数据仍不足，记录数: %d", fund_code, len(df_final))
            return df_final[['date', 'net_value']]
        else:
            if not local_df.empty and len(local_df) >= min_data_points:
                logger.info("基金 %s 使用本地数据，记录数: %d", fund_code, len(local_df))
                return local_df[['date', 'net_value']]
            else:
                logger.error("基金 %s 未获取到任何有效数据，且本地数据不足: %d 条", fund_code, len(local_df))
                raise ValueError(f"基金 {fund_code} 未获取到任何有效数据，且本地数据不足: {len(local_df)} 条")

    def _calculate_indicators(self, fund_code, df):
        """计算技术指标并生成结果字典，新增沪深300趋势信号"""
        logger.info("开始计算基金 %s 的技术指标...", fund_code)
        try:
            if df is None or df.empty or len(df) < 50: 
                logger.warning("基金 %s 数据不足，记录数: %d", fund_code, len(df) if df is not None else 0)
                return {
                    'fund_code': fund_code, 'latest_net_value': "数据不足", 'rsi': np.nan, 'ma_ratio': np.nan,
                    'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 'advice': "观察", 
                    'action_signal': 'N/A', 'ma_trend_signal': 'N/A', 'index_trend_signal': 'N/A'
                }

            df = df.sort_values(by='date', ascending=True).tail(100) 
            
            # MACD
            exp12 = df['net_value'].ewm(span=12, adjust=False).mean()
            exp26 = df['net_value'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp12 - exp26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            # 布林带 (BB)
            window = 20
            df['bb_mid'] = df['net_value'].rolling(window=window, min_periods=1).mean()
            df['bb_std'] = df['net_value'].rolling(window=window, min_periods=1).std()
            df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
            
            # RSI
            delta = df['net_value'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # MA10/MA50 趋势判断
            ma10 = df['net_value'].rolling(window=min(10, len(df)), min_periods=1).mean() 
            ma50 = df['net_value'].rolling(window=min(50, len(df)), min_periods=1).mean()
            
            latest_data = df.iloc[-1]
            latest_net_value = latest_data['net_value']
            latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan
            latest_ma10 = ma10.iloc[-1]
            latest_ma50 = ma50.iloc[-1]
            latest_ma50_ratio = latest_net_value / latest_ma50 if not pd.isna(latest_ma50) and latest_ma50 != 0 else np.nan
            latest_macd_diff = latest_data['macd'] - latest_data['signal']
            latest_bb_upper = latest_data['bb_upper']
            latest_bb_lower = latest_data['bb_lower']
            
            ma_trend_signal = "震荡"
            if latest_ma10 > latest_ma50 and latest_net_value > latest_ma10:
                ma_trend_signal = "多头趋势"
            elif latest_ma10 < latest_ma50 and latest_net_value < latest_ma50:
                ma_trend_signal = "空头趋势"

            # 新增：计算沪深300指数趋势信号
            index_trend_signal = "震荡"
            if not self.index_data.empty:
                index_df = self.index_data[self.index_data['date'] <= df['date'].max()].tail(100)
                if len(index_df) >= 50:
                    index_ma10 = index_df['net_value'].rolling(window=10, min_periods=1).mean()
                    index_ma50 = index_df['net_value'].rolling(window=50, min_periods=1).mean()
                    latest_index_ma10 = index_ma10.iloc[-1]
                    latest_index_ma50 = index_ma50.iloc[-1]
                    latest_index_value = index_df['net_value'].iloc[-1]
                    if latest_index_ma10 > latest_index_ma50 and latest_index_value > latest_index_ma10:
                        index_trend_signal = "多头趋势"
                    elif latest_index_ma10 < latest_index_ma50 and latest_index_value < latest_index_ma50:
                        index_trend_signal = "空头趋势"
                    logger.info("基金 %s 沪深300趋势: %s", fund_code, index_trend_signal)
                else:
                    logger.warning("沪深300数据不足，记录数: %d", len(index_df))

            # 投资建议逻辑（新增指数趋势条件）
            advice = "观察"
            if (not np.isnan(latest_rsi) and latest_rsi > 70) or \
               (not np.isnan(latest_bb_upper) and latest_net_value > latest_bb_upper) or \
               (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2):
                advice = "等待回调"
            elif (not np.isnan(latest_rsi) and latest_rsi < 30) or \
                 (not np.isnan(latest_bb_lower) and latest_net_value < latest_bb_lower) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.8):
                advice = "可分批买入" if index_trend_signal != "空头趋势" else "观察"
            elif (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff > 0) and \
                 index_trend_signal == "多头趋势":
                advice = "可分批买入"
            elif (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 1) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff < 0):
                advice = "等待回调"

            # 行动信号逻辑（新增指数趋势条件）
            action_signal = "持有/观察"
            if (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.95) or \
               (ma_trend_signal == "空头趋势") or \
               (index_trend_signal == "空头趋势"):
                action_signal = "强卖出/规避"
            elif (not np.isnan(latest_rsi) and latest_rsi > 70) and \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff < 0):
                action_signal = "强卖出/规避"
            elif (not np.isnan(latest_rsi) and latest_rsi > 65) or \
                 (not np.isnan(latest_bb_upper) and latest_net_value > latest_bb_upper) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio > 1.2):
                action_signal = "弱卖出/规避"
            elif (not np.isnan(latest_rsi) and latest_rsi < 35) and \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 0.9) and \
                 (not np.isnan(latest_macd_diff) and latest_macd_diff > 0) and \
                 (ma_trend_signal != "空头趋势") and \
                 (index_trend_signal != "空头趋势"):
                action_signal = "强买入"
            elif (not np.isnan(latest_rsi) and latest_rsi < 45) or \
                 (not np.isnan(latest_bb_lower) and latest_net_value < latest_bb_lower) or \
                 (not np.isnan(latest_ma50_ratio) and latest_ma50_ratio < 1):
                action_signal = "弱买入" if index_trend_signal != "空头趋势" else "持有/观察"
            
            logger.info("基金 %s 技术指标计算完成: 行动信号=%s, 投资建议=%s", fund_code, action_signal, advice)
            return {
                'fund_code': fund_code,
                'latest_net_value': latest_net_value,
                'rsi': latest_rsi,
                'ma_ratio': latest_ma50_ratio,
                'macd_diff': latest_macd_diff,
                'bb_upper': latest_bb_upper,
                'bb_lower': latest_bb_lower,
                'advice': advice,
                'action_signal': action_signal,
                'ma_trend_signal': ma_trend_signal,
                'index_trend_signal': index_trend_signal
            }

        except Exception as e:
            logger.error("处理基金 %s 时发生异常: %s", fund_code, str(e))
            return {
                'fund_code': fund_code, 'latest_net_value': "计算失败", 'rsi': np.nan,
                'ma_ratio': np.nan, 'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 
                'advice': "观察", 'action_signal': 'N/A', 'ma_trend_signal': 'N/A', 
                'index_trend_signal': 'N/A'
            }

    def get_fund_data(self):
        """主控函数：优先从本地加载，仅在数据非最新或不完整时下载"""
        logger.info("开始获取基金数据...")
        try:
            self._parse_report()
            if not self.fund_codes:
                logger.error("无有效基金代码，终止数据获取")
                return

            fund_codes_to_fetch = []
            expected_latest_date = self._get_expected_latest_date()
            min_data_points = 50

            for fund_code in self.fund_codes:
                logger.debug("处理基金 %s...", fund_code)
                local_df = self._read_local_data(fund_code)
                
                if not local_df.empty:
                    latest_local_date = local_df['date'].max().date()
                    data_points = len(local_df)
                    
                    # 修改：即使最新日期满足要求，若记录数不足50条，也触发网络请求
                    if data_points >= min_data_points and latest_local_date >= expected_latest_date:
                        logger.info("基金 %s 本地数据满足要求，跳过网络请求，记录数: %d", fund_code, data_points)
                        self.fund_data[fund_code] = self._calculate_indicators(fund_code, local_df)
                        continue
                
                fund_codes_to_fetch.append(fund_code)
            
            logger.info("需从网络获取数据的基金数: %d", len(fund_codes_to_fetch))
            if fund_codes_to_fetch:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # 修改：减少线程数以稳定
                    future_to_code = {executor.submit(self._fetch_fund_data, code): code for code in fund_codes_to_fetch}
                    for future in concurrent.futures.as_completed(future_to_code, timeout=300):  # 修改：添加5分钟超时
                        fund_code = future_to_code[future]
                        try:
                            df = future.result()
                            self.fund_data[fund_code] = self._calculate_indicators(fund_code, df)
                            logger.info("基金 %s 数据处理完成", fund_code)
                        except concurrent.futures.TimeoutError:
                            logger.error("基金 %s 处理超时", fund_code)
                            self.fund_data[fund_code] = {
                                'fund_code': fund_code, 'latest_net_value': "超时", 'rsi': np.nan,
                                'ma_ratio': np.nan, 'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 
                                'advice': "观察", 'action_signal': 'N/A', 'ma_trend_signal': 'N/A', 'index_trend_signal': 'N/A'
                            }
                        except Exception as e:
                            logger.error("获取和处理基金 %s 数据时出错: %s", fund_code, str(e))
                            self.fund_data[fund_code] = {
                                'fund_code': fund_code, 'latest_net_value': "数据获取失败", 'rsi': np.nan,
                                'ma_ratio': np.nan, 'macd_diff': np.nan, 'bb_upper': np.nan, 'bb_lower': np.nan, 
                                'advice': "观察", 'action_signal': 'N/A', 'ma_trend_signal': 'N/A', 'index_trend_signal': 'N/A'
                            }

        except Exception as e:
            logger.error("get_fund_data 执行异常: %s", e)
            raise

    def generate_report(self):
        """生成市场情绪与技术指标监控报告，新增市场趋势列"""
        logger.info("正在生成市场监控报告...")
        
        try:
            now_utc = datetime.utcnow()
            EIGHT_HOURS = timedelta(hours=8)
            now_cst = now_utc + EIGHT_HOURS
            REPORT_DATE_CST_STR = now_cst.strftime('%Y-%m-%d %H:%M:%S')
            
            report_df_list = []
            for fund_code in self.fund_codes:
                data = self.fund_data.get(fund_code)
                if data is None or data.get('latest_net_value') in ["数据获取失败", "数据不足", "超时"]: 
                    logger.debug("基金 %s 数据无效，跳过报告生成", fund_code)
                    continue

                latest_net_value_str = f"{data['latest_net_value']:.4f}" if isinstance(data['latest_net_value'], (float, int)) else str(data['latest_net_value'])
                rsi_str = f"{data['rsi']:.2f}" if isinstance(data['rsi'], (float, int)) and not np.isnan(data['rsi']) else "N/A"
                ma_ratio_str = f"{data['ma_ratio']:.2f}" if isinstance(data['ma_ratio'], (float, int)) and not np.isnan(data['ma_ratio']) else "N/A"
                
                macd_signal = "N/A"
                if isinstance(data['macd_diff'], (float, int)) and not np.isnan(data['macd_diff']):
                    macd_signal = "金叉" if data['macd_diff'] > 0 else "死叉"
                
                bollinger_pos = "中轨"
                if isinstance(data['latest_net_value'], (float, int)):
                    if isinstance(data['bb_upper'], (float, int)) and not np.isnan(data['bb_upper']) and data['latest_net_value'] > data['bb_upper']:
                        bollinger_pos = "上轨上方"
                    elif isinstance(data['bb_lower'], (float, int)) and not np.isnan(data['bb_lower']) and data['latest_net_value'] < data['bb_lower']:
                        bollinger_pos = "下轨下方"
                else:
                    bollinger_pos = "N/A"
                
                ma_trend_signal = data.get('ma_trend_signal', 'N/A')
                index_trend_signal = data.get('index_trend_signal', 'N/A')

                # 修改：修复 NameError，使用 data['rsi'] 和 data['ma_ratio']
                dca_strategy = "继续定投"
                if (not np.isnan(data['rsi']) and data['rsi'] < 40) or \
                   (not np.isnan(data['ma_ratio']) and data['ma_ratio'] < 1.0):
                    dca_strategy = "继续定投/加大买入" if index_trend_signal != "空头趋势" else "继续定投"
                elif (not np.isnan(data['rsi']) and data['rsi'] > 65) or \
                     (not np.isnan(data['ma_ratio']) and data['ma_ratio'] > 1.2):
                    dca_strategy = "继续定投/小额买入"
                elif ma_trend_signal == "空头趋势" or data['action_signal'] == "强卖出/规避":
                    dca_strategy = "暂停定投/减仓"

                report_df_list.append({
                    "基金代码": fund_code, 
                    "最新净值": latest_net_value_str, 
                    "RSI": rsi_str, 
                    "净值/MA50": ma_ratio_str,
                    "MA趋势": ma_trend_signal, 
                    "MACD信号": macd_signal, 
                    "布林带位置": bollinger_pos,
                    "市场趋势": index_trend_signal,
                    "投资建议": data['advice'], 
                    "行动信号": data['action_signal'],
                    "定投策略": dca_strategy
                })

            report_df = pd.DataFrame(report_df_list)

            order_map_action = {"强买入": 1, "弱买入": 2, "持有/观察": 3, "弱卖出/规避": 4, "强卖出/规避": 5, "N/A": 6}
            report_df['sort_order_action'] = report_df['行动信号'].map(order_map_action)
            
            report_df['RSI_num'] = pd.to_numeric(report_df['RSI'], errors='coerce')

            report_df = report_df.sort_values(
                by=['sort_order_action', 'RSI_num'],
                ascending=[True, True]
            ).drop(columns=['sort_order_action', 'RSI_num'])

            markdown_table = report_df.to_markdown(index=False)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 市场情绪与技术指标监控报告\n\n")
                f.write(f"生成日期: {REPORT_DATE_CST_STR}\n\n")
                f.write(f"## 推荐基金技术指标 (处理基金数: {len(report_df)})\n")
                f.write("此表格已按**行动信号优先级**排序，'强买入'基金将排在最前面。\n\n")
                f.write("**【定投策略（DCA）说明】**\n")
                f.write("- **继续定投/加大买入**: RSI < 40 或 净值/MA50 < 1.0，且市场趋势非空头，定投的绝佳低位。\n")
                f.write("- **继续定投/小额买入**: RSI > 65 或 净值/MA50 > 1.2，定投时应避免追高，控制份额。\n")
                f.write("- **暂停定投/减仓**: 基金自身或市场处于空头趋势，或触发强卖出信号，考虑减仓或停止新增投入。\n\n")
                f.write(markdown_table)
            
            logger.info("报告生成完成: %s", self.output_file)

        except Exception as e:
            logger.error("generate_report 执行异常: %s", e)
            raise

if __name__ == "__main__":
    try:
        logger.info("脚本启动")
        monitor = MarketMonitor()
        monitor.get_fund_data()
        monitor.generate_report()
        logger.info("脚本执行完成")
    except Exception as e:
        logger.error("脚本运行失败: %s", e)
        raise
