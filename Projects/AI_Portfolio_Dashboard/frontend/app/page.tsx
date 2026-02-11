'use client';

import { useState } from 'react';
import { apiService, PortfolioData } from '../services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Activity, DollarSign, PieChart as PieChartIcon, TrendingUp } from 'lucide-react';

// 샘플 데이터
const SAMPLE_PORTFOLIO: PortfolioData = {
  holdings: [
    { ticker: 'AAPL', quantity: 10, avg_price: 150 },
    { ticker: 'TSLA', quantity: 5, avg_price: 200 },
    { ticker: 'NVDA', quantity: 20, avg_price: 400 },
    { ticker: 'GOOGL', quantity: 15, avg_price: 120 },
  ],
};

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState<PortfolioData>(SAMPLE_PORTFOLIO);
  const [query, setQuery] = useState('');
  const [chatResponse, setChatResponse] = useState('');
  const [loading, setLoading] = useState(false);

  // 차트 데이터 변환 (간소화)
  const chartData = portfolio.holdings.map((h) => ({
    name: h.ticker,
    value: h.quantity * h.avg_price,
  }));

  const handleChat = async () => {
    if (!query) return;
    setLoading(true);
    try {
      // 포트폴리오 분석 API 호출
      const result = await apiService.analyzePortfolio(query, portfolio);
      setChatResponse(result.answer);
    } catch (error) {
      console.error('Chat Error:', error);
      setChatResponse('오류가 발생했습니다. 백엔드 서버가 실행 중인지 확인해주세요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">🏦 AI Portfolio Dashboard</h1>
        <p className="text-gray-500">Next.js & FastAPI Integration</p>
      </header>

      {/* 상단 지표 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {[
          { title: 'Total Value', value: '$25,400', icon: DollarSign, color: 'text-green-600' },
          { title: 'Profit/Loss', value: '+$3,200', icon: TrendingUp, color: 'text-blue-600' },
          { title: 'Holdings', value: '4 Stocks', icon: PieChartIcon, color: 'text-purple-600' },
          { title: 'Risk Level', value: 'Moderate', icon: Activity, color: 'text-orange-600' },
        ].map((item, idx) => (
          <Card key={idx}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{item.title}</CardTitle>
              <item.icon className={`h-4 w-4 ${item.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{item.value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* 포트폴리오 차트 */}
        <Card>
          <CardHeader>
            <CardTitle>Portfolio Composition</CardTitle>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* AI 채팅 인터페이스 */}
        <Card>
          <CardHeader>
            <CardTitle>💬 AI Financial Advisor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="h-[200px] overflow-y-auto bg-gray-100 rounded-md p-4 text-sm whitespace-pre-wrap">
                {chatResponse || '무엇이든 물어보세요! (예: "애플 주식을 더 살까?")'}
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Ask about your portfolio..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleChat()}
                />
                <Button onClick={handleChat} disabled={loading}>
                  {loading ? 'Analyzing...' : 'Send'}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
