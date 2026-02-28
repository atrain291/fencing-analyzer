import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30_000,
})

export default api
