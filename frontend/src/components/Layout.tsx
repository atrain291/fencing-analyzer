import { Outlet, NavLink } from 'react-router-dom'
import { Swords } from 'lucide-react'

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-gray-800 bg-gray-900 px-6 py-3 flex items-center gap-3">
        <Swords className="text-brand-500" size={24} />
        <span className="font-semibold text-lg tracking-tight">Fencing Analyzer</span>
        <nav className="ml-8 flex gap-4 text-sm text-gray-400">
          <NavLink
            to="/dashboard"
            className={({ isActive }) =>
              isActive ? 'text-white' : 'hover:text-white transition-colors'
            }
          >
            Dashboard
          </NavLink>
        </nav>
      </header>
      <main className="flex-1 p-6">
        <Outlet />
      </main>
    </div>
  )
}
