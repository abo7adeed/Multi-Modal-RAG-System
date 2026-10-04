import { useEffect, useRef } from 'react'

export function useAutoScroll<T extends HTMLElement>(dependency: unknown) {
  const containerRef = useRef<T | null>(null)

  useEffect(() => {
    const element = containerRef.current
    if (element?.scrollTo) {
      element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' })
    } else if (element) {
      element.scrollTop = element.scrollHeight
    }
  }, [dependency])

  return containerRef
}
