# Energy Forecast Pro - Probabilistic Demand Forecasting System

A sophisticated web-based energy demand forecasting system designed for grid operators and energy traders. The system provides any-quantile probabilistic forecasting with real-time updates and comprehensive uncertainty quantification.

## Key Features

### For Energy Grid Operators
- **Real-time demand forecasting** with 48-hour prediction horizon
- **Capacity threshold alerts** when predicted demand exceeds grid capacity
- **Historical vs. actual comparison** for forecast validation
- **Automatic updates** with live data integration

### For Energy Traders
- **Custom quantile selection** (1-99%) for risk assessment
- **Multiple quantile overlays** for comprehensive uncertainty visualization
- **Export functionality** for forecast data and reports
- **Confidence intervals** (5%, 25%, 50%, 75%, 95%)

### Time Series Management
- **Upload and manage multiple time series** datasets
- **Automatic preprocessing** (missing values, outlier detection)
- **Multi-format support** (CSV, Excel, Parquet, JSON)
- **Data quality metrics** and validation

### Domain-Specific Features
- **Temporal pattern recognition** (daily, weekly, seasonal cycles)
- **Exogenous variables integration** (weather, calendar events)
- **Peak demand management** and anomaly detection
- **Seasonality and trend detection**

## Technology Stack

- **Frontend**: React 18+ with TypeScript
- **Styling**: TailwindCSS
- **Visualizations**: Recharts for interactive time series charts
- **Animation**: Framer Motion
- **State Management**: React Hooks

## Running the Application

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

## System Architecture

The system serves energy grid operators and traders requiring reliable uncertainty quantification for operational planning and risk management. It provides:

1. **Quantile Selection Interface**: Interactive slider (1-99%) with real-time graph updates
2. **Real-time Dashboard**: Live energy demand forecasting with capacity monitoring
3. **Data Management**: Multiple time series upload and preprocessing pipeline
4. **Analytics**: Comprehensive forecasting performance metrics

## User Roles

### Primary Actors
- **Energy Grid Operators**: Real-time grid management and load balancing
- **Energy Traders**: Trading decisions based on demand forecasts

### Secondary Actors
- **Energy Planners**: Long-term infrastructure planning
- **System Administrators**: Infrastructure and system maintenance

## License

This project is based on the SaaS Dashboard Design available at https://www.figma.com/design/RGTCuuqKw8BwrrR3GV6lTl/SaaS-Dashboard-Design
