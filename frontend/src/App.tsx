import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import ProcessingStatus from '@/pages/ProcessingStatus'
import VideoReview from '@/pages/VideoReview'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/bouts/:boutId/processing" element={<ProcessingStatus />} />
          <Route path="/bouts/:boutId/review" element={<VideoReview />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
