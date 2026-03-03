import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import ConfigureROI from '@/pages/ConfigureROI'
import ProcessingStatus from '@/pages/ProcessingStatus'
import VideoReview from '@/pages/VideoReview'
import DrillReport from '@/pages/DrillReport'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/bouts/:boutId/configure" element={<ConfigureROI />} />
          <Route path="/bouts/:boutId/processing" element={<ProcessingStatus />} />
          <Route path="/bouts/:boutId/review" element={<VideoReview />} />
          <Route path="/bouts/:boutId/drill" element={<DrillReport />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
