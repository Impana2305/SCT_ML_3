import { useEffect, useRef } from 'react'
import './ParticlesBackground.css'

const PARTICLE_COUNT = 25

function ParticlesBackground() {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const particles = []

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const particle = document.createElement('div')
      particle.className = 'particle'

      const size = Math.random() * 4 + 2
      const x = Math.random() * 100
      const duration = Math.random() * 15 + 10
      const delay = Math.random() * 15
      const hue = Math.random() > 0.5 ? '270' : '320'

      particle.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${x}%;
        animation-duration: ${duration}s;
        animation-delay: -${delay}s;
        background: hsl(${hue}, 80%, 65%);
        box-shadow: 0 0 ${size * 3}px hsl(${hue}, 80%, 65%);
      `
      container.appendChild(particle)
      particles.push(particle)
    }

    return () => {
      particles.forEach(p => p.remove())
    }
  }, [])

  return (
    <>
      <div ref={containerRef} className="particles-container" aria-hidden="true" />
      <div className="glow-orb orb-1" aria-hidden="true" />
      <div className="glow-orb orb-2" aria-hidden="true" />
      <div className="glow-orb orb-3" aria-hidden="true" />
    </>
  )
}

export default ParticlesBackground
