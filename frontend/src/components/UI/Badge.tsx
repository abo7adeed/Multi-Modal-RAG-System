import type { ReactNode } from 'react'

type BadgeVariant = 'neutral' | 'accent' | 'success' | 'danger'

const VARIANTS: Record<BadgeVariant, string> = {
  neutral: 'bg-surface-2 text-muted',
  accent: 'bg-accent-soft text-accent-text',
  success: 'bg-surface-2 text-success',
  danger: 'bg-danger-soft text-danger-text',
}

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
}

export function Badge({ children, variant = 'neutral' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium tracking-wide ${VARIANTS[variant]}`}
    >
      {children}
    </span>
  )
}
