import { ShieldIcon } from './Icons'

export type Route = 'analyze' | 'history' | 'stats'

const LINKS: { route: Route; label: string }[] = [
  { route: 'analyze', label: 'Analyze' },
  { route: 'history', label: 'History' },
  { route: 'stats', label: 'Stats' },
]

export default function Header({ route }: { route: Route }) {
  return (
    <header className="bg-navy-900 text-white">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <a href="#/analyze" className="flex items-center gap-2.5 rounded">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-navy-700">
            <ShieldIcon style={{ color: 'var(--risk-safe)' }} width={22} height={22} />
          </span>
          <span>
            <span className="block text-lg font-bold leading-tight">TriGuard</span>
            <span className="block text-[11px] text-navy-200">Verify before you apply</span>
          </span>
        </a>

        <nav aria-label="Main">
          <ul className="flex flex-wrap gap-1">
            {LINKS.map((l) => (
              <li key={l.route}>
                <a
                  href={`#/${l.route}`}
                  aria-current={route === l.route ? 'page' : undefined}
                  className={`block rounded-md px-3 py-1.5 text-sm font-medium ${
                    route === l.route ? 'bg-navy-700 text-white' : 'text-navy-100 hover:bg-navy-800'
                  }`}
                >
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  )
}
