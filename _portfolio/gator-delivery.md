---
title: "GatorDelivery: Priority-Based Order Management System"
excerpt: "Dual AVL tree implementation for efficient delivery order scheduling with dynamic priority and ETA tracking"
collection: portfolio
date: 2024-04-15
permalink: /portfolio/gator-delivery
---

## Overview
GatorDelivery is a sophisticated order management system designed for single-driver delivery services, implementing dual AVL trees to simultaneously optimize order priority and estimated time of arrival (ETA). The system handles real-time order creation, cancellation, and ETA updates while maintaining balanced tree structures for efficient query operations.

[View on GitHub](https://github.com/skapoor2024/ads_gator_delievery)

## Key Features

 **Dual AVL Tree Architecture**
- Separate trees for priority-based and time-based ordering
- Automatic rebalancing after insertions and deletions
- O(log n) complexity for core operations
- Custom comparison functions for flexible ordering

 **Dynamic Order Management**
- Real-time priority calculation based on order value and delivery time
- ETA updates with cascading tree restructuring
- Order cancellation with synchronized tree cleanup
- Rank queries for order position tracking

 **Efficient Query Operations**
- Range queries for orders within time windows
- Predecessor/successor finding for delivery scheduling
- Bulk operations using tree traversal algorithms
- Hash map integration for O(1) order lookup

## Technical Implementation

### Core Components
**AVL Tree Structure**: Self-balancing binary search tree with height-based rotation logic. Implements left and right rotations to maintain balance factor within [-1, 1] range.

**Priority System**: Combines order value and delivery time with configurable weights (`value_wt`, `time_wt`) to compute dynamic priorities using normalization constant.

**ETA Tree**: Maintains orders sorted by estimated delivery time, enabling efficient time-based range queries and schedule optimization.

**Hash Map Integration**: Unordered map stores order ID to order object mapping, providing constant-time access while trees maintain sorted views.

### Key Algorithms
- **Insertion**: Insert into both trees with O(log n) rebalancing
- **Deletion**: Remove from priority tree, ETA tree, and hash map atomically
- **Update ETA**: Delete from ETA tree, modify timestamp, reinsert with updated position
- **Get Rank**: Traverse priority tree counting nodes with higher priority
- **Range Query**: In-order traversal with bounds checking

## Experimental Results

### Performance Characteristics
- **Tree Height**: Maintained at O(log n) through AVL balancing
- **Insertion Time**: Consistent O(log n) performance across 1000+ orders
- **Query Efficiency**: Sub-millisecond response for rank and range queries
- **Memory Usage**: Linear space complexity with dual tree overhead

### Operational Constraints
- Single delivery person model
- Sequential order fulfillment (one at a time)
- Centralized pickup location
- Real-time priority recalculation on ETA changes

## Technical Highlights

**C++ Design Patterns**: Smart pointers (`std::shared_ptr`) for automatic memory management, function objects for custom comparators, and modular class hierarchy.

**Balanced Tree Operations**: Implements complete AVL rotation logic (LL, RR, LR, RL cases) with height tracking and balance factor computation.

**Dual Indexing Strategy**: Maintains consistency between priority-ordered and time-ordered views through synchronized insertions and deletions.

**Edge Case Handling**: Manages empty tree scenarios, single-node trees, and cascading deletions when ETA updates affect tree structure.

**Technologies**: C++17, AVL Trees, Hash Maps, CMake, Template Programming
