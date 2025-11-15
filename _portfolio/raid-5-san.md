---
title: "RAID-5 Storage Area Network"
excerpt: "Fault-tolerant distributed storage system with parity-based recovery and client-server architecture"
collection: portfolio
date: 2024-03-20
permalink: /portfolio/raid-5-san
---

## Overview
RAID-5 SAN is a distributed, fault-tolerant storage system built on a Python-based UNIX file system using client/server architecture with Remote Procedure Calls (RPC). The system addresses critical storage challenges including uneven load distribution, soft failures (data degradation), and hard failures (server shutdown) through RAID-5 configuration with distributed parity and checksum-based recovery.

[View on GitHub](https://github.com/skapoor2024/RAID-5_SAN)

## Key Features

 **RAID-5 Architecture**
- Distributed parity across multiple block servers
- XOR-based parity computation for data recovery
- Dynamic virtual-to-physical block mapping
- Checksum validation using MD5 for corruption detection

 **Fault Tolerance & Recovery**
- Automatic detection of corrupted blocks via checksum mismatches
- Real-time recovery from single server failures
- Server repair functionality to rebuild failed disk contents
- Graceful handling of soft failures (data degradation) and hard failures (server crashes)

 **Client-Server Model**
- XML-RPC protocol for network communication
- Multiple concurrent clients with cache invalidation
- Lock-based synchronization using Read-and-Set-Memory (RSM)
- Block-level caching with write-through policy

## Technical Implementation

### Core Components
**DiskBlocks Client**: Manages virtual block layer with automatic mapping to physical servers. Implements acquire/release locking, cache management, and transparent fault recovery.

**Block Servers**: Independent XML-RPC servers storing raw blocks with MD5 checksums. Handle Get/Put operations with corruption detection and configurable delay simulation.

**Virtual-to-Physical Mapping**: Distributes blocks across N servers using modulo arithmetic, with parity rotation to balance load. For N servers and virtual block V: data server = (V mod (N-1)), parity server rotates anti-clockwise.

**Recovery Mechanism**: XORs all blocks in a stripe (excluding failed server) to reconstruct missing data or parity. Works for single-server failures in RAID-5 configuration.

### Key Algorithms
- **Put Operation**: Update data block and parity using XOR: new_parity = old_parity XOR old_data XOR new_data
- **Get Operation**: Check cache, validate checksum, recover if corrupted using stripe XOR
- **Server Repair**: Iterate all blocks, reconstruct each using RecoverBlock(), restore to repaired server
- **Lock Protocol**: RSM block for mutual exclusion, last-writer tracking for cache coherence

## Experimental Results

### Fault Tolerance Validation
- **Single Server Failure**: Successful data recovery with <10% performance degradation
- **Corrupted Block Detection**: 100% accuracy in identifying and repairing corrupted data via checksums
- **Load Distribution**: Average load variance <15% across servers under uniform workload

### Performance Characteristics
- **Cache Hit Rate**: 70-85% for sequential access patterns with write-through caching
- **Recovery Overhead**: 2-3x latency during active recovery compared to normal operations
- **RPC Latency**: ~5-10ms per block operation over localhost XML-RPC

### System Resilience
Successfully handled simulated failures including server timeouts, block corruption, and cascading invalidations across multiple clients.

## Technical Highlights

**XOR-Based Parity**: Efficient recovery using bitwise XOR operations on byte arrays, enabling single-disk fault tolerance without full data replication.

**Checksum Validation**: MD5 hashing detects silent data corruption before serving blocks, triggering automatic recovery pipeline.

**Cache Coherence Protocol**: Last-writer tracking with client ID ensures cache invalidation when different clients modify shared blocks.

**Distributed Lock Management**: RSM protocol using dedicated block for lock acquisition prevents race conditions in multi-client scenarios.

**Technologies**: Python 3, XML-RPC, RAID-5, MD5 Checksums, Socket Programming, Distributed Systems
