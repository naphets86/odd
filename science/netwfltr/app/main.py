from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
import databases
import sqlalchemy
from datetime import datetime, date
from typing import Optional
import subprocess
import re
import os

DATABASE_URL = "sqlite:////data/netwatch.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

traffic_table = sqlalchemy.Table(
    "traffic",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("protocol", sqlalchemy.String(32)),
    sqlalchemy.Column("src_ip", sqlalchemy.String(64)),
    sqlalchemy.Column("dst_ip", sqlalchemy.String(64)),
    sqlalchemy.Column("src_port", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("dst_port", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("length", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("flags", sqlalchemy.String(64), nullable=True),
    sqlalchemy.Column("info", sqlalchemy.String(512), nullable=True),
    sqlalchemy.Column("osi_layer", sqlalchemy.Integer, nullable=True),
)

engine = sqlalchemy.create_engine(DATABASE_URL.replace("+aiosqlite", ""))

PROTOCOL_OSI_MAP = {
    # ── Layer 2 - Data Link ───────────────────────────────────────────────
    "ARP": 2, "RARP": 2, "LLDP": 2,
    # EtherCAT läuft direkt auf Ethernet (EtherType 0x88A4)
    "ETHERCAT": 2, "ECAT": 2,
    # PROFINET DCP/LLDP-basierte Discovery
    "PROFINET": 2, "PN-DCP": 2, "PN-PTCP": 2,
    # SERCOS III (EtherType 0x88CD)
    "SERCOS": 2,
    # POWERLINK (EtherType 0x88AB)
    "POWERLINK": 2, "EPL": 2,
    # CDP (Cisco Discovery Protocol)
    "CDP": 2,

    # ── Layer 3 - Network ─────────────────────────────────────────────────
    "IP": 3, "IPv4": 3, "IPv6": 3, "ICMP": 3, "ICMPv6": 3, "IGMP": 3,
    "OSPF": 3, "BGP": 3, "VRRP": 3, "PIM": 3,
    "GRE": 3, "IPIP": 3,

    # ── Layer 4 - Transport ───────────────────────────────────────────────
    "TCP": 4, "UDP": 4, "SCTP": 4,

    # ── Layer 5 - Session ─────────────────────────────────────────────────
    "RPC": 5, "NetBIOS": 5, "PPTP": 5,
    "OPCUA": 5, "OPC-UA": 5,

    # ── Layer 6 - Presentation ────────────────────────────────────────────
    "TLS": 6, "SSL": 6, "SSH": 6, "DTLS": 6,

    # ── Layer 7 - Application (Standard IT) ──────────────────────────────
    "HTTP": 7, "HTTPS": 7, "DNS": 7, "FTP": 7, "SMTP": 7, "IMAP": 7,
    "POP3": 7, "DHCP": 7, "NTP": 7, "SNMP": 7, "LDAP": 7, "SIP": 7,
    "RTP": 7, "QUIC": 7, "HTTP2": 7, "MDNS": 7, "LLMNR": 7,
    "TFTP": 7, "SFTP": 7, "TELNET": 7, "SYSLOG": 7, "RADIUS": 7,
    "MQTT": 7, "AMQP": 7, "XMPP": 7, "WEBSOCKET": 7,

    # ── Layer 7 - Industrieautomation & Feldbus ───────────────────────────
    # Modbus TCP (Port 502)
    "MODBUS": 7, "MODBUS-TCP": 7,
    # EtherNet/IP / CIP (TCP 44818, UDP 2222)
    "ETHERNET/IP": 7, "ENIP": 7, "CIP": 7,
    # PROFINET IO (Applikationsschicht)
    "PROFINET-IO": 7, "PNIO": 7,
    # OPC UA (TCP 4840, HTTPS 4843)
    "OPC-UA-TCP": 7, "OPC-UA-HTTPS": 7,
    # BACnet (UDP 47808) — Gebäudeautomation
    "BACNET": 7, "BACNET-IP": 7,
    # DNP3 (TCP/UDP 20000) — Energieversorgung/SCADA
    "DNP3": 7,
    # IEC 60870-5-104 (TCP 2404) — Energieautomation
    "IEC104": 7, "IEC-104": 7,
    # IEC 61850 / GOOSE / MMS (TCP 102)
    "IEC61850": 7, "GOOSE": 7, "MMS": 7, "SAMPLED-VALUES": 7, "SV": 7,
    # HART-IP (UDP/TCP 5094)
    "HART-IP": 7, "HART": 7,
    # IO-Link
    "IO-LINK": 7,
    # CoAP — IoT/M2M (UDP 5683, DTLS 5684)
    "COAP": 7,
    # FINS — Omron (UDP/TCP 9600)
    "FINS": 7,
    # S7comm — Siemens S7 PLC (TCP 102)
    "S7COMM": 7, "S7PLUS": 7,
    # MTConnect — CNC-Maschinenüberwachung (HTTP 5000)
    "MTCONNECT": 7,
    # EtherCAT AoE / ADS (TCP 48898)
    "AOE": 7, "ADS": 7,
    # PROFIBUS over TCP (Gateway)
    "PROFIBUS": 7,
    # CANopen over Ethernet
    "CANOPEN": 7,
    # DeviceNet over IP
    "DEVICENET": 7,
    # SECS/GEM — Halbleiterfertigung
    "SECS": 7, "GEM": 7, "HSMS": 7,
    # Omron SYSMAC
    "SYSMAC": 7,
    # Yokogawa Vnet/IP
    "VNET": 7,
}

def get_osi_layer(protocol: str) -> int:
    if not protocol:
        return 3
    p = protocol.upper().split("/")[0].strip()
    return PROTOCOL_OSI_MAP.get(p, 3)

capture_task = None

async def run_capture():
    """Run tcpdump and parse output into DB"""
    try:
        process = await asyncio.create_subprocess_exec(
            "tcpdump", "-l", "-n", "-q", "-i", "any",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            entry = parse_tcpdump_line(line.decode("utf-8", errors="replace").strip())
            if entry:
                await database.execute(traffic_table.insert().values(**entry))
    except Exception as e:
        print(f"Capture error: {e}")
        await asyncio.sleep(5)

def parse_tcpdump_line(line: str) -> Optional[dict]:
    if not line or len(line) < 10:
        return None
    try:
        # Timestamp pattern: HH:MM:SS.ffffff
        ts_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d+)\s+(.*)", line)
        if not ts_match:
            return None
        time_str, rest = ts_match.groups()
        now = datetime.utcnow()
        try:
            t = datetime.strptime(time_str.split(".")[0], "%H:%M:%S")
            ts = now.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
        except:
            ts = now

        protocol = "IP"
        src_ip, dst_ip = "", ""
        src_port, dst_port = None, None
        flags = None
        info = rest[:300]

        # ARP
        if "ARP," in rest or rest.startswith("ARP"):
            protocol = "ARP"
            arp_m = re.search(r"(\d+\.\d+\.\d+\.\d+).*?(\d+\.\d+\.\d+\.\d+)", rest)
            if arp_m:
                src_ip, dst_ip = arp_m.group(1), arp_m.group(2)
        # IP based
        elif "IP6" in rest:
            protocol = "IPv6"
            ip6_m = re.search(r"(\S+?)\.?(\d+)?\s+>\s+(\S+?)\.?(\d+)?,", rest)
            if ip6_m:
                src_ip = ip6_m.group(1)
                src_port = int(ip6_m.group(2)) if ip6_m.group(2) else None
                dst_ip = ip6_m.group(3)
                dst_port = int(ip6_m.group(4)) if ip6_m.group(4) else None
        elif "IP " in rest:
            ip_m = re.search(r"IP\s+(\S+)\s+>\s+(\S+)[:,]", rest)
            if ip_m:
                src_full = ip_m.group(1)
                dst_full = ip_m.group(2)
                src_parts = src_full.rsplit(".", 1)
                dst_parts = dst_full.rsplit(".", 1)
                src_ip = src_parts[0]
                dst_ip = dst_parts[0]
                try:
                    src_port = int(src_parts[1]) if len(src_parts) > 1 else None
                except:
                    src_port = None
                try:
                    dst_port = int(dst_parts[1]) if len(dst_parts) > 1 else None
                except:
                    dst_port = None

            # Detect protocol from ports / keywords
            if dst_port == 443 or src_port == 443:
                protocol = "TLS"
            elif dst_port == 80 or src_port == 80:
                protocol = "HTTP"
            elif dst_port == 53 or src_port == 53:
                protocol = "DNS"
            elif dst_port == 22 or src_port == 22:
                protocol = "SSH"
            elif dst_port == 25 or src_port == 25:
                protocol = "SMTP"
            elif dst_port in (67, 68) or src_port in (67, 68):
                protocol = "DHCP"
            elif dst_port == 123 or src_port == 123:
                protocol = "NTP"
            elif dst_port == 161 or src_port == 161:
                protocol = "SNMP"
            elif dst_port == 21 or src_port == 21:
                protocol = "FTP"
            elif dst_port == 23 or src_port == 23:
                protocol = "TELNET"
            elif dst_port == 143 or src_port == 143:
                protocol = "IMAP"
            elif dst_port == 110 or src_port == 110:
                protocol = "POP3"
            elif dst_port in (389, 636) or src_port in (389, 636):
                protocol = "LDAP"
            elif dst_port in (5060, 5061) or src_port in (5060, 5061):
                protocol = "SIP"
            # ── Industrieprotokolle ──────────────────────────────────
            elif dst_port == 502 or src_port == 502:
                protocol = "MODBUS"
            elif dst_port in (44818, 2222) or src_port in (44818, 2222):
                protocol = "ENIP"
            elif dst_port in (4840, 4843) or src_port in (4840, 4843):
                protocol = "OPC-UA-TCP"
            elif dst_port == 47808 or src_port == 47808:
                protocol = "BACNET"
            elif dst_port == 20000 or src_port == 20000:
                protocol = "DNP3"
            elif dst_port == 2404 or src_port == 2404:
                protocol = "IEC104"
            elif dst_port == 102 or src_port == 102:
                protocol = "S7COMM"
            elif dst_port == 5094 or src_port == 5094:
                protocol = "HART-IP"
            elif dst_port == 9600 or src_port == 9600:
                protocol = "FINS"
            elif dst_port == 48898 or src_port == 48898:
                protocol = "AOE"
            elif dst_port == 1883 or src_port == 1883:
                protocol = "MQTT"
            elif dst_port == 8883 or src_port == 8883:
                protocol = "MQTT"
            elif dst_port in (5683, 5684) or src_port in (5683, 5684):
                protocol = "COAP"
            elif dst_port == 5000 or src_port == 5000:
                protocol = "MTCONNECT"
            elif "UDP" in rest:
                protocol = "UDP"
            elif "tcp" in rest.lower() or "TCP" in rest:
                protocol = "TCP"
                tcp_m = re.search(r"Flags \[([^\]]+)\]", rest)
                if tcp_m:
                    flags = tcp_m.group(1)
            elif "ICMP" in rest:
                protocol = "ICMP"

        if not src_ip:
            return None

        length = None
        len_m = re.search(r"length (\d+)", rest)
        if len_m:
            length = int(len_m.group(1))

        return {
            "timestamp": ts,
            "protocol": protocol,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "length": length,
            "flags": flags,
            "info": info[:300],
            "osi_layer": get_osi_layer(protocol),
        }
    except Exception as e:
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("/data", exist_ok=True)
    metadata.create_all(engine)
    await database.connect()
    global capture_task
    capture_task = asyncio.create_task(run_capture())
    yield
    if capture_task:
        capture_task.cancel()
    await database.disconnect()

app = FastAPI(title="Network Filter", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
templates = Jinja2Templates(directory="/app/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/traffic")
async def get_traffic(
    protocol: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    time_from: Optional[str] = Query(None),
    time_to: Optional[str] = Query(None),
    limit: int = Query(500),
    offset: int = Query(0),
):
    query = traffic_table.select().order_by(traffic_table.c.timestamp.desc())
    conditions = []

    if protocol and protocol != "ALL":
        conditions.append(traffic_table.c.protocol == protocol)
    if date_from:
        conditions.append(traffic_table.c.timestamp >= f"{date_from} 00:00:00")
    if date_to:
        conditions.append(traffic_table.c.timestamp <= f"{date_to} 23:59:59")
    if time_from:
        conditions.append(sqlalchemy.func.strftime("%H:%M:%S", traffic_table.c.timestamp) >= time_from)
    if time_to:
        conditions.append(sqlalchemy.func.strftime("%H:%M:%S", traffic_table.c.timestamp) <= time_to)

    if conditions:
        query = query.where(sqlalchemy.and_(*conditions))

    total_q = sqlalchemy.select(sqlalchemy.func.count()).select_from(traffic_table)
    if conditions:
        total_q = total_q.where(sqlalchemy.and_(*conditions))
    total = await database.fetch_val(total_q)

    query = query.limit(limit).offset(offset)
    rows = await database.fetch_all(query)
    return {
        "total": total,
        "data": [dict(r) for r in rows]
    }

@app.get("/api/protocols")
async def get_protocols():
    query = sqlalchemy.select(traffic_table.c.protocol, sqlalchemy.func.count().label("count"))\
        .group_by(traffic_table.c.protocol).order_by(sqlalchemy.desc("count"))
    rows = await database.fetch_all(query)
    return [dict(r) for r in rows]

@app.get("/api/stats")
async def get_stats():
    total = await database.fetch_val(sqlalchemy.select(sqlalchemy.func.count()).select_from(traffic_table))
    layer_q = sqlalchemy.select(
        traffic_table.c.osi_layer,
        sqlalchemy.func.count().label("count")
    ).group_by(traffic_table.c.osi_layer).order_by(traffic_table.c.osi_layer)
    layers = await database.fetch_all(layer_q)
    return {"total": total, "by_layer": [dict(r) for r in layers]}

@app.delete("/api/traffic")
async def clear_traffic():
    await database.execute(traffic_table.delete())
    return {"status": "cleared"}