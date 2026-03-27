# Energy Forecast Pro - Probabilistic Demand Forecasting System

A sophisticated web-based energy demand forecasting system designed for grid operators and energy traders. The system provides any-quantile probabilistic forecasting with real-time updates and comprehensive uncertainty quantification.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Languages & Libraries](#languages--libraries)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [User Roles](#user-roles)
- [License](#license)

## Overview

Energy Forecast Pro is a modern SaaS application that combines advanced energy forecasting capabilities with an intuitive user interface. Built with cutting-edge web technologies, it enables energy professionals to make data-driven decisions based on probabilistic forecasts with confidence intervals and custom quantile analysis.

## Key Features

### For Energy Grid Operators

- **Real-time demand forecasting** with 48-hour prediction horizon
- **Capacity threshold alerts** when predicted demand exceeds grid capacity
- **Historical vs. actual comparison** for forecast validation
- **Automatic updates** with live data integration
- **Demand alert system** with configurable thresholds and notifications

### For Energy Traders

- **Custom quantile selection** (1-99% range) for risk assessment
- **Multiple quantile overlays** for comprehensive uncertainty visualization
- **Export functionality** for forecast data and reports
- **Confidence intervals** (5%, 25%, 50%, 75%, 95%)
- **Advanced visualization** with interactive charts and drill-down capabilities

### Time Series Management

- **Upload and manage multiple time series** datasets
- **Automatic preprocessing** (missing values, outlier detection)
- **Multi-format support** (CSV, Excel, Parquet, JSON)
- **Data quality metrics** and validation
- **Batch data processing** with progress tracking

### Domain-Specific Features

- **Temporal pattern recognition** (daily, weekly, seasonal cycles)
- **Exogenous variables integration** (weather, calendar events)
- **Peak demand management** and anomaly detection
- **Seasonality and trend detection**
- **Forecast composition analysis**

### Interactive Components

- **Dashboard**: Real-time overview of forecasts and metrics
- **Analytics Page**: In-depth analysis and performance metrics
- **Forecast Page**: Detailed quantile visualization and exploration
- **Data Management Page**: Upload, preview, and manage datasets
- **Settings Page**: User preferences and system configuration
- **Command Palette**: Quick navigation and command execution

## Technology Stack

### Frontend Framework

- **React 18.3.1** - UI library with hooks-based architecture
- **TypeScript** - Type-safe development
- **Vite 6.3.5** - Fast build tool and development server
- **React DOM 18.3.1** - React rendering for web

### Styling & UI

- **TailwindCSS 4.1.12** - Utility-first CSS framework
- **@mui/material 7.3.5** - Material Design components
- **@mui/icons-material 7.3.5** - Material Design icons
- **@emotion/react & @emotion/styled 11.14.x** - CSS-in-JS styling
- **Radix UI** - Unstyled, accessible component library (full suite):
  - Accordion, Alert Dialog, Aspect Ratio, Avatar
  - Checkbox, Collapsible, Context Menu, Dialog, Dropdown Menu
  - Hover Card, Label, Menubar, Navigation Menu, Popover
  - Progress, Radio Group, Scroll Area, Select, Separator
  - Slider, Switch, Tabs, Toggle, Tooltip, and more
- **lucide-react 0.487.0** - Beautiful and consistent icon library
- **tailwind-merge 3.2.0** - Merge Tailwind CSS classes intelligently

### Charts & Visualizations

- **Recharts 2.15.2** - Composable React charting library
- **embla-carousel-react 8.6.0** - Carousel/slider library
- **react-resizable-panels 2.1.7** - Resizable panel layouts

### Form & Input Handling

- **react-hook-form 7.55.0** - Performant form library
- **cmdk 1.1.1** - Command menu component
- **input-otp 1.4.2** - OTP input component
- **react-day-picker 8.10.1** - Date picker component
- **date-fns 3.6.0** - Modern date utility library

### Animation & Motion

- **motion 12.23.24** - Animation library
- **tw-animate-css 1.3.8** - Tailwind CSS animations
- **next-themes 0.4.6** - Theme management

### Utilities

- **react-dnd 16.0.1** - Drag and drop functionality
- **react-dnd-html5-backend 16.0.1** - HTML5 drag-drop backend
- **react-popper 2.3.0** - Popper positioning engine
- **react-slick 0.31.0** - Carousel component
- **react-responsive-masonry 2.7.1** - Masonry layout
- **clsx 2.1.1** - Utility for conditional classNames
- **class-variance-authority 0.7.1** - CVA for component variants
- **vaul 1.1.2** - Drawer component
- **sonner 2.0.3** - Toast notification library
- **@popperjs/core 2.11.8** - Positioning engine

## Languages & Libraries

### Programming Languages

- **TypeScript** - Primary development language for type safety and better tooling
- **JavaScript/ES6+** - React components and utilities
- **JSX/TSX** - React component syntax

### Package Manager

- **npm** - Node Package Manager
- **pnpm** - Alternative package manager support (with version overrides for Vite)

### Build Tools

- **Vite** - Next-generation frontend tooling
- **PostCSS** - CSS transformations (config: postcss.config.mjs)
- **@tailwindcss/vite** - Tailwind CSS integration for Vite

### Development Dependencies

- **@vitejs/plugin-react 4.7.0** - React plugin for Vite
- **tailwindcss 4.1.12** - CSS framework
- **vite 6.3.5** - Build tool

## Project Structure

```
src/
├── app/
│   ├── components/              # React components
│   │   ├── AnalyticsPage.tsx    # Analytics dashboard
│   │   ├── AuthPage.tsx         # Authentication interface
│   │   ├── CommandPalette.tsx   # Command palette for navigation
│   │   ├── Dashboard.tsx        # Main dashboard view
│   │   ├── DataPage.tsx         # Data management page
│   │   ├── ForecastPage.tsx     # Forecast visualization page
│   │   ├── Header.tsx           # Application header
│   │   ├── SettingsPage.tsx     # User settings
│   │   ├── Sidebar.tsx          # Navigation sidebar
│   │   ├── GlassCard.tsx        # Glass-morphism card component
│   │   ├── MetricCard.tsx       # Metric display component
│   │   ├── StatCard.tsx         # Statistics card
│   │   ├── EmptyState.tsx       # Empty state UI
│   │   ├── LoadingSkeleton.tsx  # Loading placeholder
│   │   ├── PremiumButton.tsx    # Premium feature button
│   │   ├── Toast.tsx            # Toast notifications
│   │   └── ui/                  # Shared UI components
│   ├── context/                 # React context providers
│   └── App.tsx                  # Root component
├── styles/                      # Global styles
└── main.tsx                     # Application entry point

backend/                         # Backend service directory
config/                          # Configuration files
guidelines/                      # Development guidelines
```

## Installation & Setup

### Prerequisites

- Node.js 18.x or higher
- npm or pnpm package manager

### Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd Saas-main
```

2. **Install dependencies**

```bash
npm install
# or
pnpm install
```

3. **Start the development server**

```bash
npm run dev
# or
pnpm dev
```

The application will be available at `http://localhost:5173` (or similar, as Vite will indicate)

4. **Build for production**

```bash
npm run build
# or
pnpm build
```

### Environment Configuration

- Copy `.env.example` to `.env.local` if available
- Configure API endpoints and other settings as needed

## Usage

### Main Features

1. **Dashboard**
   - View real-time energy demand forecasts
   - Monitor capacity thresholds and alerts
   - Access key performance metrics

2. **Forecast Page**
   - Interactive quantile selection (1-99%)
   - Multiple confidence intervals visualization
   - Export forecast data

3. **Data Management**
   - Upload time series datasets
   - Preview and validate data
   - Configure preprocessing options

4. **Analytics**
   - Performance metrics and KPIs
   - Historical forecast accuracy
   - Trend analysis and pattern recognition

5. **Settings**
   - User preferences and configuration
   - Theme selection (light/dark mode via next-themes)
   - Alert thresholds and notification settings

### Keyboard Shortcuts

- **Cmd/Ctrl + K** - Open Command Palette for quick navigation

## System Architecture

The system serves energy grid operators and traders requiring reliable uncertainty quantification for operational planning and risk management. It provides:

1. **Quantile Selection Interface**: Interactive slider (1-99%) with real-time graph updates
2. **Real-time Dashboard**: Live energy demand forecasting with capacity monitoring
3. **Data Management**: Multiple time series upload and preprocessing pipeline
4. **Analytics**: Comprehensive forecasting performance metrics
5. **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

### Component Architecture

- **Page Components**: High-level page layouts (Dashboard, Analytics, Forecast, etc.)
- **UI Components**: Reusable UI elements from Radix UI and custom components
- **Context Providers**: Application state management using React Context
- **Utility Components**: Helpers for common functionality (cards, skeletons, etc.)

## User Roles

### Primary Actors

- **Energy Grid Operators**: Real-time grid management and load balancing
- **Energy Traders**: Trading decisions based on demand forecasts

### Secondary Actors

- **Energy Planners**: Long-term infrastructure planning
- **System Administrators**: Infrastructure and system maintenance

## Browser Support

The application supports all modern browsers including:

- Chrome/Chromium (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance Optimizations

- **Vite**: Lightning-fast build and development server
- **React 18**: Concurrent rendering and automatic batching
- **TailwindCSS**: Optimized CSS with tree-shaking
- **Code Splitting**: Automatic route-based code splitting

## Security Features

- **TypeScript**: Type safety reduces runtime errors
- **Form Validation**: Client-side validation with react-hook-form
- **HTTPS Ready**: Production-ready security practices
- **XSS Protection**: React's built-in protection against XSS attacks

## Documentation

- [Getting Started Guide](./GETTING_STARTED.md)
- [Quick Reference](./QUICK_REFERENCE.md)
- [Architecture Documentation](./ARCHITECTURE.md)
- [Backend API Requirements](./BACKEND_API_REQUIREMENTS.md)
- [Technical Reference](./TECHNICAL_REFERENCE.md)
- [Testing Guide](./TESTING_GUIDE.md)

## License

This project is based on the SaaS Dashboard Design available at https://www.figma.com/design/RGTCuuqKw8BwrrR3GV6lTl/SaaS-Dashboard-Design

## Contributing

Contributions are welcome! Please ensure:

- TypeScript types are properly defined
- Components follow React best practices
- CSS classes use TailwindCSS utilities
- Code is well-documented

## Support & Questions

For issues, feature requests, or questions, please refer to the documentation files or contact the development team.

---

**Last Updated**: March 2026
**Version**: 1.0.0
