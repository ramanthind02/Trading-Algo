#!/usr/bin/env python3
"""
Demo: Interactive Brokers Market Data Fetching
==============================================

Consolidated demo script for fetching market data from Interactive Brokers API.
Demonstrates:
1. Requesting head timestamp (earliest available data)
2. Requesting historical bar data
3. Requesting live/streaming market data

Prerequisites:
- IBKR TWS or IB Gateway running on localhost
- Market data subscriptions (or use delayed data for testing)
- ibapi installed (not via pip - must use official IBKR installation)

Run: python scripts/demo_ib_data_fetch.py

Author: Trading Research Team
Date: 2026-01-18
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import datetime
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IB API imports (must be installed via IBKR official installation)
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.ticktype import TickTypeEnum
    from ibapi.common import OrderId, BarData
except ImportError as e:
    print(f"Error: IB API not found. {e}")
    print("Please install IB API following official IBKR instructions:")
    print("https://www.interactivebrokers.com/campus/ibkr-api-page/trader-workstation-api/")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class IBConfig:
    """Immutable configuration for IB connection."""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS Paper: 7497, TWS Live: 7496, IB Gateway Paper: 4002, IB Gateway Live: 4001
    client_id: int = 0


class MarketDataType(Enum):
    """Market data type enumeration."""
    LIVE = 1
    FROZEN = 2
    DELAYED = 3
    DELAYED_FROZEN = 4


@dataclass(frozen=True)
class ContractSpec:
    """Immutable contract specification."""
    symbol: str
    sec_type: str = "STK"  # STK, FUT, OPT, CASH, etc.
    exchange: str = "SMART"
    currency: str = "USD"


# =============================================================================
# IB API CLIENT
# =============================================================================

class IBDataClient(EClient, EWrapper):
    """
    Interactive Brokers API client for fetching market data.
    
    Combines EClient (outgoing requests) and EWrapper (incoming callbacks).
    """
    
    def __init__(self, config: IBConfig):
        """Initialize the IB API client."""
        EClient.__init__(self, self)
        self.config = config
        self.order_id: int = 0
        self.connected: bool = False
        self.historical_bars: List[BarData] = []
        self.head_timestamp: Optional[str] = None
        self.historical_data_complete: dict[int, bool] = {}  # Track completion by reqId
        self.historical_data_error: dict[int, str] = {}  # Track errors by reqId
        
    def nextValidId(self, orderId: OrderId) -> None:
        """Callback when connection is established and valid order ID is received."""
        self.order_id = orderId
        self.connected = True
        print(f"[OK] Connected to IB. Next valid order ID: {self.order_id}")
    
    def nextId(self) -> int:
        """Get next request ID."""
        self.order_id += 1
        return self.order_id
    
    def error(self, *args) -> None:
        """
        Error callback handler (supports both old and new API signatures).
        
        Old signature: error(reqId, errorCode, errorString, advancedOrderReject="")
        New signature: error(reqId, errorTime, errorCode, errorString, advancedOrderReject="")
        """
        # Determine which signature based on number of args
        if len(args) == 3:
            # Old signature: reqId, errorCode, errorString
            reqId, errorCode, errorString = args
            advancedOrderReject = ""
            errorTime = ""
        elif len(args) == 4:
            # Could be old with advancedOrderReject or new without it
            # Check if second arg looks like a timestamp (string) or error code (int)
            if isinstance(args[1], str) and not args[1].isdigit():
                # New signature: reqId, errorTime, errorCode, errorString
                reqId, errorTime, errorCode, errorString = args
                advancedOrderReject = ""
            else:
                # Old signature: reqId, errorCode, errorString, advancedOrderReject
                reqId, errorCode, errorString, advancedOrderReject = args
                errorTime = ""
        elif len(args) == 5:
            # New signature: reqId, errorTime, errorCode, errorString, advancedOrderReject
            reqId, errorTime, errorCode, errorString, advancedOrderReject = args
        else:
            # Unexpected number of arguments
            print(f"ERROR: Unexpected error signature with {len(args)} arguments: {args}")
            return
        
        # Filter out informational messages (but log them for debugging)
        if errorCode in (2104, 2106, 2158):
            # These are informational connection messages, not errors
            return

        # Error 2176 is just a warning about fractional shares - not an actual error
        if errorCode == 2176:
            return

        # Error 366 "No historical data query found" is expected after we cancel
        # the request in historicalDataEnd - ignore it if data was already received
        if errorCode == 366 and self.historical_data_complete.get(reqId, False):
            return

        # Store error for historical data requests (only if not already complete)
        if reqId > 0 and not self.historical_data_complete.get(reqId, False):
            self.historical_data_error[reqId] = f"Code {errorCode}: {errorString}"
            # Mark as complete even on error so waiting loops can exit
            self.historical_data_complete[reqId] = True
        
        error_msg = f"ERROR reqId: {reqId}, Code: {errorCode}, Message: {errorString}"
        if errorTime:
            error_msg += f", Time: {errorTime}"
        print(error_msg)
        if advancedOrderReject:
            print(f"  Advanced Reject: {advancedOrderReject}")
    
    # ========================================================================
    # HEAD TIMESTAMP METHODS
    # ========================================================================
    
    def headTimestamp(
        self, 
        reqId: int, 
        headTimeStamp: str
    ) -> None:
        """
        Callback for head timestamp request.
        
        Returns the earliest available timestamp for historical data.
        """
        self.head_timestamp = headTimeStamp
        print(f"\n{'='*60}")
        print(f"Head Timestamp (reqId: {reqId})")
        print(f"{'='*60}")
        print(f"Epoch timestamp: {headTimeStamp}")
        
        try:
            dt = datetime.datetime.fromtimestamp(int(headTimeStamp))
            print(f"Human readable: {dt}")
        except (ValueError, OSError) as e:
            print(f"Could not convert timestamp: {e}")
        
        # Cancel the request
        self.cancelHeadTimeStamp(reqId)
    
    def request_head_timestamp(
        self, 
        contract: Contract, 
        what_to_show: str = "TRADES",
        use_rth: int = 1,
        format_date: int = 2
    ) -> None:
        """
        Request the earliest available timestamp for a contract.
        
        Parameters
        ----------
        contract : Contract
            The contract to query
        what_to_show : str
            Data type: "TRADES", "MIDPOINT", "BID", "ASK", etc.
        use_rth : int
            1 = regular trading hours only, 0 = include extended hours
        format_date : int
            1 = UTC string format, 2 = Epoch timestamp (integer)
        """
        req_id = self.nextId()
        self.reqHeadTimeStamp(req_id, contract, what_to_show, use_rth, format_date)
    
    # ========================================================================
    # HISTORICAL DATA METHODS
    # ========================================================================
    
    def historicalData(
        self, 
        reqId: int, 
        bar: BarData
    ) -> None:
        """
        Callback for historical bar data.
        
        Each bar is received individually.
        """
        self.historical_bars.append(bar)
        print(
            f"Bar (reqId: {reqId}): "
            f"Date: {bar.date}, "
            f"Open: {bar.open:.2f}, "
            f"High: {bar.high:.2f}, "
            f"Low: {bar.low:.2f}, "
            f"Close: {bar.close:.2f}, "
            f"Volume: {bar.volume}"
        )
    
    def historicalDataEnd(
        self, 
        reqId: int, 
        start: str, 
        end: str
    ) -> None:
        """
        Callback when all historical data has been received.
        
        Parameters
        ----------
        reqId : int
            Request ID
        start : str
            Start timestamp of the data range
        end : str
            End timestamp of the data range
        """
        # Mark this request as complete
        self.historical_data_complete[reqId] = True
        
        print(f"\n{'='*60}")
        print(f"Historical Data Complete (reqId: {reqId})")
        print(f"{'='*60}")
        print(f"Received {len(self.historical_bars)} bars")
        print(f"Start: {start}")
        print(f"End: {end}")
        
        # Cancel the request
        self.cancelHistoricalData(reqId)
    
    def request_historical_data(
        self,
        contract: Contract,
        end_date_time: str = "",
        duration: str = "1 D",
        bar_size: str = "1 hour",
        what_to_show: str = "TRADES",
        use_rth: int = 1,
        format_date: int = 1,
        keep_up_to_date: bool = False
    ) -> int:
        """
        Request historical bar data.
        
        Parameters
        ----------
        contract : Contract
            The contract to query
        end_date_time : str
            End date/time in format "YYYYMMDD HH:MM:SS TZ" or "" for current time
        duration : str
            Duration string (e.g., "1 D", "1 W", "1 M", "1 Y")
        bar_size : str
            Bar size (e.g., "1 sec", "5 secs", "1 min", "1 hour", "1 day")
        what_to_show : str
            Data type: "TRADES", "MIDPOINT", "BID", "ASK", etc.
        use_rth : int
            1 = regular trading hours only, 0 = include extended hours
        format_date : int
            1 = UTC string format, 2 = Epoch timestamp
        keep_up_to_date : bool
            If True, continue updating bars as new data arrives
        
        Returns
        -------
        int
            Request ID for this historical data request
        """
        req_id = self.nextId()
        self.historical_bars = []  # Reset for new request
        self.historical_data_complete[req_id] = False  # Mark as in progress
        self.historical_data_error[req_id] = ""  # Clear any previous errors
        
        self.reqHistoricalData(
            req_id,
            contract,
            end_date_time,
            duration,
            bar_size,
            what_to_show,
            use_rth,
            format_date,
            keep_up_to_date,
            []
        )
        
        return req_id
    
    def wait_for_historical_data(
        self,
        req_id: int,
        timeout: float = 30.0,
        poll_interval: float = 0.1
    ) -> bool:
        """
        Wait for historical data request to complete.
        
        Parameters
        ----------
        req_id : int
            Request ID to wait for
        timeout : float
            Maximum time to wait in seconds
        poll_interval : float
            How often to check for completion in seconds
        
        Returns
        -------
        bool
            True if data was received, False if timeout or error
        """
        start_time = time.time()
        while not self.historical_data_complete.get(req_id, False):
            if time.time() - start_time > timeout:
                print(f"\n[WARN] Timeout waiting for historical data (reqId: {req_id})")
                return False
            time.sleep(poll_interval)
        
        # Check for errors
        if req_id in self.historical_data_error and self.historical_data_error[req_id]:
            print(f"\n[WARN] Error received for historical data (reqId: {req_id}): {self.historical_data_error[req_id]}")
            return False
        
        return True
    
    # ========================================================================
    # LIVE MARKET DATA METHODS
    # ========================================================================
    
    def tickPrice(
        self, 
        reqId: int, 
        tickType: int, 
        price: float, 
        attrib: object
    ) -> None:
        """
        Callback for price-related market data ticks.
        
        Parameters
        ----------
        reqId : int
            Request ID
        tickType : int
            Tick type (see TickTypeEnum)
        price : float
            Price value
        attrib : object
            Additional attributes
        """
        tick_name = TickTypeEnum.toStr(tickType)
        print(f"Price Tick (reqId: {reqId}): {tick_name} = {price}")
    
    def tickSize(
        self, 
        reqId: int, 
        tickType: int, 
        size: int
    ) -> None:
        """
        Callback for size-related market data ticks.
        
        Parameters
        ----------
        reqId : int
            Request ID
        tickType : int
            Tick type (see TickTypeEnum)
        size : int
            Size value
        """
        tick_name = TickTypeEnum.toStr(tickType)
        print(f"Size Tick (reqId: {reqId}): {tick_name} = {size}")
    
    def request_market_data(
        self,
        contract: Contract,
        generic_tick_list: str = "232",  # Mark price
        snapshot: bool = False,
        regulatory_snapshot: bool = False
    ) -> int:
        """
        Request live/streaming market data.
        
        Parameters
        ----------
        contract : Contract
            The contract to query
        generic_tick_list : str
            Comma-separated generic tick types (e.g., "232,233,234")
        snapshot : bool
            If True, return single snapshot (last 11 seconds aggregated)
        regulatory_snapshot : bool
            If True, return regulatory snapshot (costs ~$0.01 per request)
        
        Returns
        -------
        int
            Request ID for this market data request
        """
        req_id = self.nextId()
        self.reqMktData(
            req_id,
            contract,
            generic_tick_list,
            snapshot,
            regulatory_snapshot,
            []
        )
        return req_id
    
    def cancel_market_data(self, req_id: int) -> None:
        """Cancel a market data subscription."""
        self.cancelMktData(req_id)
    
    def set_market_data_type(self, data_type: MarketDataType) -> None:
        """
        Set the market data type (Live, Delayed, etc.).
        
        Parameters
        ----------
        data_type : MarketDataType
            Type of market data to request
        """
        self.reqMarketDataType(data_type.value)
    
    # ========================================================================
    # CONNECTION METHODS
    # ========================================================================
    
    def connect_to_ib(self) -> None:
        """Connect to IB TWS/Gateway."""
        print(f"Connecting to IB at {self.config.host}:{self.config.port}...")
        EClient.connect(self, self.config.host, self.config.port, self.config.client_id)
        
        # Start message processing thread
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        
        # Wait for connection
        time.sleep(1)
        
        if not self.connected:
            print("Warning: Connection may not be established. Check TWS/Gateway is running.")
    
    def disconnect_from_ib(self) -> None:
        """Disconnect from IB."""
        if self.connected:
            EClient.disconnect(self)
            self.connected = False
            print("Disconnected from IB")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_contract(spec: ContractSpec) -> Contract:
    """
    Create an IB Contract from specification.
    
    Parameters
    ----------
    spec : ContractSpec
        Contract specification
        
    Returns
    -------
    Contract
        IB API Contract object
    """
    contract = Contract()
    contract.symbol = spec.symbol
    contract.secType = spec.sec_type
    contract.exchange = spec.exchange
    contract.currency = spec.currency
    return contract


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_head_timestamp(client: IBDataClient) -> None:
    """Demonstrate requesting head timestamp."""
    print("\n" + "="*80)
    print("DEMO 1: Requesting Head Timestamp")
    print("="*80)
    
    contract_spec = ContractSpec(
        symbol="AAPL",
        sec_type="STK",
        exchange="SMART",
        currency="USD"
    )
    contract = create_contract(contract_spec)
    
    client.request_head_timestamp(
        contract,
        what_to_show="TRADES",
        use_rth=1,
        format_date=2  # Epoch timestamp
    )
    
    # Wait for response
    time.sleep(2)


def demo_historical_data(client: IBDataClient) -> None:
    """Demonstrate requesting historical bar data."""
    print("\n" + "="*80)
    print("DEMO 2: Requesting Historical Bar Data")
    print("="*80)
    
    contract_spec = ContractSpec(
        symbol="AAPL",
        sec_type="STK",
        exchange="SMART",
        currency="USD"
    )
    contract = create_contract(contract_spec)
    
    # Request 1 day of hourly bars
    end_time = "20240523 16:00:00 US/Eastern"
    
    req_id = client.request_historical_data(
        contract,
        end_date_time=end_time,
        duration="1 D",
        bar_size="1 hour",
        what_to_show="TRADES",
        use_rth=1,
        format_date=1,
        keep_up_to_date=False
    )
    
    print(f"Requested historical data with reqId: {req_id}")
    print("Waiting for data...")
    
    # Wait for data to arrive
    time.sleep(5)


def demo_live_market_data(client: IBDataClient) -> None:
    """Demonstrate requesting live/streaming market data."""
    print("\n" + "="*80)
    print("DEMO 3: Requesting Live Market Data")
    print("="*80)
    
    # Set to delayed data (for testing without subscription)
    client.set_market_data_type(MarketDataType.DELAYED)
    print("Using DELAYED market data (15-minute delay)")
    print("Change to MarketDataType.LIVE for real-time data (requires subscription)")
    
    contract_spec = ContractSpec(
        symbol="AAPL",
        sec_type="STK",
        exchange="SMART",
        currency="USD"
    )
    contract = create_contract(contract_spec)
    
    req_id = client.request_market_data(
        contract,
        generic_tick_list="232",  # Mark price
        snapshot=False,
        regulatory_snapshot=False
    )
    
    print(f"Subscribed to market data with reqId: {req_id}")
    print("Streaming data for 10 seconds...")
    print("(Press Ctrl+C to stop early)\n")
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping early...")
    
    # Cancel subscription
    client.cancel_market_data(req_id)
    print(f"\nCancelled market data subscription (reqId: {req_id})")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main demo function."""
    print("="*80)
    print("Interactive Brokers Market Data Fetching Demo")
    print("="*80)
    print("\nPrerequisites:")
    print("  1. TWS or IB Gateway must be running")
    print("  2. API settings enabled in TWS/Gateway")
    print("  3. Market data subscriptions (or use delayed data)")
    print("\nNote: This demo uses delayed data by default to avoid subscription requirements.")
    print("="*80)
    
    # Configuration
    config = IBConfig(
        host="127.0.0.1",
        port=7497,  # Adjust for your setup
        client_id=0
    )
    
    # Create client
    client = IBDataClient(config)
    
    try:
        # Connect
        client.connect_to_ib()
        
        # Wait for connection to establish
        max_wait = 5
        waited = 0
        while not client.connected and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        
        if not client.connected:
            print("\nERROR: Could not establish connection to IB.")
            print("Please ensure:")
            print("  1. TWS or IB Gateway is running")
            print("  2. API is enabled in TWS/Gateway settings")
            print("  3. Port matches your configuration")
            return
        
        # Run demos
        demo_head_timestamp(client)
        time.sleep(1)
        
        demo_historical_data(client)
        time.sleep(1)
        
        demo_live_market_data(client)
        
        print("\n" + "="*80)
        print("Demo complete!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if client.connected:
            client.disconnect_from_ib()


if __name__ == "__main__":
    main()
