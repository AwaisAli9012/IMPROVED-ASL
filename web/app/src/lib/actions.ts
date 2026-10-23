/** Adds `.seen` once the element scrolls into view, so CSS can run a one-time entrance. */
export function reveal(node: HTMLElement, delay = 0) {
  node.style.setProperty('--delay', `${delay}ms`);
  const io = new IntersectionObserver(([e]) => {
    if (e.isIntersecting) { node.classList.add('seen'); io.disconnect(); }
  }, { threshold: 0.2 });
  io.observe(node);
  return { destroy: () => io.disconnect() };
}
