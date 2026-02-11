import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// API 클라이언트 인스턴스 생성
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface PortfolioData {
    holdings: Array<{
        ticker: string;
        quantity: number;
        avg_price: number;
    }>;
}

export interface AnalysisResponse {
    answer: string;
    tool_calls: string[];
}

export const apiService = {
    // 포트폴리오 분석 요청
    analyzePortfolio: async (query: string, portfolio: PortfolioData) => {
        const response = await apiClient.post<AnalysisResponse>('/portfolio/analyze', {
            query,
            portfolio,
        });
        return response.data;
    },

    // 시장 리서치 요청
    researchMarket: async (query: string) => {
        const response = await apiClient.post<AnalysisResponse>('/market/research', {
            query,
        });
        return response.data;
    },

    // 헬스 체크
    checkHealth: async () => {
        try {
            const response = await axios.get('http://localhost:8000/');
            return response.data;
        } catch (error) {
            return null;
        }
    },
};
