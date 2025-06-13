// animations.js

window.addEventListener('load', () => {
  const canvas = document.getElementById('animationCanvas');
  const ctx = canvas.getContext('2d');

  // Set canvas dimensions to full window size
  function resizeCanvas() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Particle class definition
  class Particle {
      constructor() {
          this.x = Math.random() * canvas.width;
          this.y = Math.random() * canvas.height;
          this.size = Math.random() * 3 + 1;
          this.speedX = (Math.random() - 0.5) * 2;
          this.speedY = (Math.random() - 0.5) * 2;
          this.color = 'rgba(187, 134, 252, 0.7)'; // soft purple
      }

      update() {
          this.x += this.speedX;
          this.y += this.speedY;

          // Reverse direction upon reaching canvas edges
          if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
          if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
      }

      draw() {
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
          ctx.fillStyle = this.color;
          ctx.fill();
      }
  }

  // Create an array of particles
  let particles = [];
  const particleCount = 100;
  function initParticles() {
      particles = [];
      for (let i = 0; i < particleCount; i++) {
          particles.push(new Particle());
      }
  }
  initParticles();

  // Track mouse position for interactive effect
  const mouse = { x: undefined, y: undefined, radius: 100 };
  window.addEventListener('mousemove', (event) => {
      mouse.x = event.x;
      mouse.y = event.y;
  });

  // Animation loop
  function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle, i) => {
          particle.update();
          particle.draw();

          // Apply a simple interaction: attract particles toward the mouse pointer
          const dx = mouse.x - particle.x;
          const dy = mouse.y - particle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < mouse.radius) {
              const angle = Math.atan2(dy, dx);
              const force = (mouse.radius - distance) / mouse.radius;
              particle.x += Math.cos(angle) * force * 2;
              particle.y += Math.sin(angle) * force * 2;
          }

          // Optionally, connect nearby particles with a light line
          for (let j = i + 1; j < particles.length; j++) {
              const dx = particle.x - particles[j].x;
              const dy = particle.y - particles[j].y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 100) {
                  ctx.beginPath();
                  ctx.strokeStyle = 'rgba(187, 134, 252, 0.1)';
                  ctx.lineWidth = 1;
                  ctx.moveTo(particle.x, particle.y);
                  ctx.lineTo(particles[j].x, particles[j].y);
                  ctx.stroke();
              }
          }
      });

      requestAnimationFrame(animate);
  }
  animate();
});
