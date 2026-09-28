"""
Pydantic Data Models for Delta Exchange interactions and Market Data.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProductInfo(BaseModel):
    id: int
    symbol: str
    contract_type: Optional[str] = None
    tick_size: Optional[float] = None
    contract_value: Optional[float] = None
    state: Optional[str] = None
    underlying_asset: Optional[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None


class TickerData(BaseModel):
    symbol: str
    mark_price: float
    spot_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    timestamp: int


class CandleData(BaseModel):
    symbol: str
    resolution: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBookData(BaseModel):
    symbol: str
    timestamp: Optional[int] = None
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)


class TradeData(BaseModel):
    trade_id: Optional[str] = None
    symbol: str
    price: float
    size: float
    side: str
    timestamp: int


class AccountBalance(BaseModel):
    asset_id: Optional[int] = None
    asset_symbol: str
    balance: float
    available_balance: float
    reserved_balance: float = 0.0


class PositionInfo(BaseModel):
    symbol: str
    product_id: Optional[int] = None
    side: str  # 'long', 'short', 'flat'
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class OrderInfo(BaseModel):
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: str
    order_type: str
    price: Optional[float] = None
    size: float
    status: str
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    created_at: Optional[str] = None


class ExchangeHealthReport(BaseModel):
    status: str  # 'healthy', 'degraded', 'unreachable'
    base_url: str
    reachable: bool
    latency_ms: Optional[float] = None
    auth_configured: bool
    xrp_symbol_configured: str
    xrp_symbol_valid: bool
    available_xrp_products: List[str] = Field(default_factory=list)
    server_time: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
