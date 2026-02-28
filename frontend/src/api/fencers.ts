import api from './client'

export interface Fencer {
  id: number
  name: string
  created_at: string
}

export async function listFencers(): Promise<Fencer[]> {
  const { data } = await api.get<Fencer[]>('/fencers/')
  return data
}

export async function createFencer(name: string): Promise<Fencer> {
  const { data } = await api.post<Fencer>('/fencers/', { name })
  return data
}
