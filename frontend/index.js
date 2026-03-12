class ChartManager {
  constructor() {
    this.chart = null;
    this.candleseries = null;
    this.candles = [];
    this.currentTicker = null;
    this.currentInterval = null;
    this.hasMore = false;
    this.earliestTimestamp = null;
    this.htf = ['H4', 'D', 'W', 'M'];
    this.kibotSeries = null;
    this.kibotVisible = false;
    this.domElement = document.getElementById('tvchart');

    this.initializeChart();
    this._initTickerSelector();
    this._initLoadMore();
    this._initDateRange();
    this._initKibotToggle();
  }

  initializeChart() {
    this.chart = LightweightCharts.createChart(this.domElement, {
      layout: { background: { color: '#D2D3DF' } },
      timeScale: { timeVisible: true, secondsVisible: true },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
    });
    this.candleseries = this.chart.addCandlestickSeries();
  }

  // ── Ticker selector ───────────────────────────────────────────────

  async _initTickerSelector() {
    const resp = await fetch('/tickers');
    const tickers = await resp.json();
    const select = document.getElementById('ticker_selector');

    tickers.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      select.appendChild(opt);
    });

    const defaultTicker = tickers.includes('ES') ? 'ES' : tickers[0];
    if (defaultTicker) {
      select.value = defaultTicker;
      this._loadTicker(defaultTicker);
    }

    select.addEventListener('change', () => this._loadTicker(select.value));
  }

  async _loadTicker(ticker) {
    this.currentTicker = ticker;
    this.candles = [];
    this._clearDateRange();
    this._removeKibotSeries();
    this.kibotVisible = false;

    const resp = await fetch(`/timeframes/${ticker}`);
    const timeframes = await resp.json();
    this._buildSwitcher(timeframes);
    this._resetKibot();

    await this._fetchAndRender();
  }

  // ── Timeframe switcher ────────────────────────────────────────────

  _buildSwitcher(timeframes) {
    const container = document.getElementById('timeframe_switcher');
    container.innerHTML = '';
    this._intervals = timeframes;

    const defaultTf = timeframes.includes('D') ? 'D' : timeframes[timeframes.length - 1];
    this.currentInterval = defaultTf;

    this._intervalElements = timeframes.map(tf => {
      const btn = document.createElement('button');
      btn.textContent = tf;
      btn.classList.add('switcher-item');
      if (tf === defaultTf) btn.classList.add('switcher-active-item');
      btn.addEventListener('click', () => this._onTfClick(tf));
      container.appendChild(btn);
      return btn;
    });
  }

  _onTfClick(tf) {
    if (tf === this.currentInterval) return;

    this._intervalElements.forEach((el, i) => {
      el.classList.toggle('switcher-active-item', this._intervals[i] === tf);
    });
    this.currentInterval = tf;
    this.candles = [];
    this._clearDateRange();
    this._resetKibot();
    this._fetchAndRender();
  }

  // ── Data fetching ─────────────────────────────────────────────────

  async fetchCandles(ticker, timeframe, opts = {}) {
    const params = new URLSearchParams();

    if (opts.from) params.set('from', opts.from);
    if (opts.to) params.set('to', opts.to);
    if (opts.before) params.set('before', opts.before);
    if (opts.count) params.set('count', opts.count);

    const qs = params.toString();
    const url = `/candles/${ticker}/${timeframe}${qs ? '?' + qs : ''}`;
    const resp = await fetch(url);
    return resp.json();
  }

  async _fetchAndRender() {
    const dateFrom = document.getElementById('date_from').value.trim();
    const dateTo = document.getElementById('date_to').value.trim();
    const warning = document.getElementById('cap_warning');
    warning.textContent = '';

    let opts = {};
    if (dateFrom || dateTo) {
      if (dateFrom) opts.from = dateFrom;
      if (dateTo) opts.to = dateTo;
    }

    const result = await this.fetchCandles(this.currentTicker, this.currentInterval, opts);
    this.candles = result.candles;
    this.hasMore = result.has_more;
    this.earliestTimestamp = result.earliest_timestamp;

    if (result.capped) {
      warning.textContent = `Showing last ${result.cap.toLocaleString()} of ${result.total_available.toLocaleString()} candles in range`;
    }

    this._updateLoadMoreButton();
    this.syncToInterval(this.currentInterval);
  }

  // ── Load More ─────────────────────────────────────────────────────

  _initLoadMore() {
    const btn = document.getElementById('load_more_btn');
    btn.addEventListener('click', () => this._loadMore());
  }

  async _loadMore() {
    if (!this.hasMore || !this.earliestTimestamp) return;

    const btn = document.getElementById('load_more_btn');
    btn.disabled = true;
    btn.textContent = 'Loading…';

    const result = await this.fetchCandles(
      this.currentTicker,
      this.currentInterval,
      { before: this.earliestTimestamp }
    );

    // Prepend older candles
    this.candles = [...result.candles, ...this.candles];
    this.hasMore = result.has_more;
    this.earliestTimestamp = result.earliest_timestamp;

    btn.disabled = false;
    btn.textContent = 'Load 500 More';
    this._updateLoadMoreButton();
    this.syncToInterval(this.currentInterval);
  }

  _updateLoadMoreButton() {
    const btn = document.getElementById('load_more_btn');
    btn.style.display = this.hasMore ? 'inline-block' : 'none';
  }

  // ── Date range filter ─────────────────────────────────────────────

  _initDateRange() {
    document.getElementById('date_go_btn').addEventListener('click', () => {
      this.candles = [];
      this._fetchAndRender();
    });

    document.getElementById('date_clear_btn').addEventListener('click', () => {
      this._clearDateRange();
      this.candles = [];
      this._fetchAndRender();
    });
  }

  _clearDateRange() {
    document.getElementById('date_from').value = '';
    document.getElementById('date_to').value = '';
    document.getElementById('cap_warning').textContent = '';
  }

  // ── Kibot comparison overlay ──────────────────────────────────

  _kibotTimeframes = ['D', 'W', 'M'];

  _initKibotToggle() {
    const btn = document.getElementById('kibot_toggle_btn');
    if (!btn) return;
    btn.addEventListener('click', () => this._toggleKibot());
  }

  async _resetKibot() {
    this._removeKibotSeries();
    this.kibotVisible = false;
    const btn = document.getElementById('kibot_toggle_btn');
    if (!btn) return;
    btn.textContent = 'Compare Kibot';
    btn.classList.remove('kibot-active');

    if (!this._kibotTimeframes.includes(this.currentInterval)) {
      btn.style.display = 'none';
      return;
    }

    // Probe server to check if Kibot comparison data exists for this ticker
    try {
      const resp = await fetch(`/candles/${this.currentTicker}/${this.currentInterval}/kibot?count=1`);
      btn.style.display = resp.ok ? 'inline-block' : 'none';
    } catch {
      btn.style.display = 'none';
    }
  }

  async _toggleKibot() {
    const btn = document.getElementById('kibot_toggle_btn');
    if (this.kibotVisible) {
      this._removeKibotSeries();
      btn.textContent = 'Compare Kibot';
      btn.classList.remove('kibot-active');
      this.kibotVisible = false;
      return;
    }

    btn.disabled = true;
    btn.textContent = 'Loading…';

    const params = new URLSearchParams();
    params.set('count', '5000');
    const qs = params.toString();
    const url = `/candles/${this.currentTicker}/${this.currentInterval}/kibot?${qs}`;

    try {
      const resp = await fetch(url);
      if (!resp.ok) {
        btn.textContent = 'No Kibot data';
        btn.disabled = false;
        return;
      }
      const data = await resp.json();

      this.kibotSeries = this.chart.addLineSeries({
        color: 'orange',
        lineWidth: 2,
      });
      this.kibotSeries.setData(data.candles);

      this.kibotVisible = true;
      btn.textContent = 'Hide Kibot';
      btn.classList.add('kibot-active');
    } catch {
      btn.textContent = 'Compare Kibot';
    }
    btn.disabled = false;
  }

  _removeKibotSeries() {
    if (this.kibotSeries) {
      this.chart.removeSeries(this.kibotSeries);
      this.kibotSeries = null;
    }
  }

  // ── Chart rendering ───────────────────────────────────────────────

  syncToInterval(interval) {
    if (this.candleseries) {
      this.chart.removeSeries(this.candleseries);
      this.candleseries = null;
    }

    this.candleseries = this.chart.addCandlestickSeries({
      upColor: 'rgb(30,192,53)',
      downColor: 'rgb(234,57,67)',
      wickUpColor: 'rgb(30,192,53)',
      wickDownColor: 'rgb(234,57,67)',
      borderVisible: false,
    });

    if (this.candles && this.candles.length > 0) {
      this.candleseries.setData(this.candles);
    }

    this.drawDailyDividers(interval);
    this.drawWeeklyDividers(interval);
  }

  // ── Dividers ──────────────────────────────────────────────────────

  drawDailyDividers(tf) {
    if (this.htf.includes(tf)) return;
    if (!this.candles) return;

    let prevDay = null;
    this.candles.forEach(item => {
      const date = new Date(item.time * 1000);
      const day = date.getUTCDate();
      const hour = date.getUTCHours();

      if (prevDay !== null && day !== prevDay && hour === 0) {
        this._drawVline(item.time, 'rgb(128, 128, 128)', 1);
      }
      prevDay = day;
    });
  }

  drawWeeklyDividers(tf) {
    if (this.htf.includes(tf)) return;
    if (!this.candles) return;

    let prevWeek = null;
    this.candles.forEach(item => {
      const date = new Date(item.time * 1000);
      const week = this._getWeekNumber(date);
      if (prevWeek !== null && week !== prevWeek) {
        this._drawVline(item.time, 'rgb(0, 0, 0)', 3);
      }
      prevWeek = week;
    });
  }

  _getWeekNumber(date) {
    const firstDay = new Date(date.getFullYear(), 0, 1);
    const pastDays = (date - firstDay) / 86400000;
    return Math.ceil((pastDays + firstDay.getDay() + 1) / 7);
  }

  _drawVline(time, color, width = 2) {
    const vline = new VertLine(this.chart, this.candleseries, time, { color, width });
    this.candleseries.attachPrimitive(vline);
  }
}

// ── Bootstrap ─────────────────────────────────────────────────────

const manager = new ChartManager();
