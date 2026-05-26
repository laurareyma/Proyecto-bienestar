/* ============================================
   BIENESTAR UNISABANA — Lógica de la aplicación
   ============================================ */

/* =============================================
   ESTADO GLOBAL
   ============================================= */
const state = {
  name: '',
  semester: '',
  answers: [],        // valores 1, 2 o 3 por pregunta
  currentQuestion: 0,
};

/* =============================================
   DATOS DEL QUIZ
   ============================================= */
const questions = [
  // --- Adaptación académica ---
  {
    category: 'Adaptación académica',
    text: '¿Cómo describes tu asistencia a clases en las últimas semanas?',
    options: [
      { label: 'Asisto a todas o casi todas',              value: 1 },
      { label: 'He faltado a algunas sin razón clara',     value: 2 },
      { label: 'He faltado a muchas y me cuesta retomar',  value: 3 },
    ],
  },
  {
    category: 'Adaptación académica',
    text: '¿Cómo te sientes con la carga académica?',
    options: [
      { label: 'La manejo bien',                           value: 1 },
      { label: 'A veces me desborda pero lo llevo',        value: 2 },
      { label: 'Siento que no puedo con todo',             value: 3 },
    ],
  },
  {
    category: 'Adaptación académica',
    text: '¿Cómo va tu desempeño en evaluaciones?',
    options: [
      { label: 'Bien, entrego y me va como espero',                   value: 1 },
      { label: 'He perdido algunas o entregado tarde',                value: 2 },
      { label: 'He perdido varias materias o estoy en riesgo',        value: 3 },
    ],
  },
  // --- Bienestar emocional ---
  {
    category: 'Bienestar emocional',
    text: '¿Cómo describes tu estado emocional en las últimas dos semanas?',
    options: [
      { label: 'Me siento bien en general',                           value: 1 },
      { label: 'He estado ansioso/a o triste con frecuencia',         value: 2 },
      { label: 'Me siento muy mal y me cuesta funcionar',             value: 3 },
    ],
  },
  {
    category: 'Bienestar emocional',
    text: '¿Cómo es tu calidad de sueño?',
    options: [
      { label: 'Duermo bien y descanso',                  value: 1 },
      { label: 'Tengo noches malas o irregulares',        value: 2 },
      { label: 'Duermo muy mal o casi no duermo',         value: 3 },
    ],
  },
  {
    category: 'Bienestar emocional',
    text: '¿Tienes personas con quienes hablar cuando tienes dificultades?',
    options: [
      { label: 'Sí, cuento con familia o amigos',         value: 1 },
      { label: 'A veces, pero me cuesta pedir ayuda',     value: 2 },
      { label: 'Casi no tengo a nadie',                   value: 3 },
    ],
  },
  // --- Vida universitaria ---
  {
    category: 'Vida universitaria',
    text: '¿Cómo ha sido tu adaptación a la vida universitaria?',
    options: [
      { label: 'Me he adaptado bien',                     value: 1 },
      { label: 'Ha sido difícil pero avanzo',             value: 2 },
      { label: 'Me siento perdido/a o desconectado/a',    value: 3 },
    ],
  },
  {
    category: 'Vida universitaria',
    text: '¿Has pensado en abandonar la universidad?',
    options: [
      { label: 'No, estoy motivado/a',                    value: 1 },
      { label: 'Lo he pensado pero no en serio',          value: 2 },
      { label: 'Sí, lo he pensado seriamente',            value: 3 },
    ],
  },
];

/* =============================================
   RECURSOS POR NIVEL DE RIESGO
   ============================================= */
const resources = {
  verde: [
    { icon: '🎯', text: 'CREA: talleres de habilidades y autogestión del aprendizaje' },
    { icon: '⚽', text: 'Deporte y cultura: actividades para conectar con la comunidad universitaria' },
    { icon: '📚', text: 'Portal de recursos académicos en línea de Unisabana' },
  ],
  amarillo: [
    { icon: '🤝', text: 'Tutorías entre pares CREA: apoyo académico de estudiantes avanzados' },
    { icon: '💙', text: 'Tu Línea Amiga — Orientación psicológica: (091) 861-5555 ext. 20223' },
    { icon: '🏥', text: 'Centro Médico Unisabana: atención en salud física y emocional' },
  ],
  rojo: [
    { icon: '🆘', text: 'Tu Línea Amiga: (091) 861-5555 ext. 20223 · tulineamiga20@unisabana.edu.co · Edificio F sin cita' },
    { icon: '🎓', text: 'Acompañamiento CREA: seguimiento personalizado de tu proceso académico' },
    { icon: '🏥', text: 'Centro Médico: atención inmediata disponible en campus' },
  ],
};

/* =============================================
   FUNCIÓN DE CLASIFICACIÓN — RANDOM FOREST
   -----------------------------------------------
   Delega a clasificarRF(), definida en model.js.
   model.js es generado por train_model.py.

   Para actualizar el modelo con datos reales:
     1. Agrega tus datos a train_model.py (sección DATOS)
     2. Ejecuta: python train_model.py
     3. El nuevo model.js reemplaza al anterior automáticamente.

   Si model.js no está disponible, la función cae al
   clasificador de reglas como respaldo.
   ============================================= */
function clasificar(respuestas) {
  // Usar el Random Forest exportado si está disponible
  if (typeof clasificarRF === 'function') {
    return clasificarRF(respuestas);
  }

  // Respaldo: regla heurística original (sin modelo cargado)
  const total = respuestas.reduce((a, b) => a + b, 0);
  const q8 = respuestas[7];
  if (q8 === 3) return 'rojo';
  if (total <= 11) return 'verde';
  if (total <= 18) return 'amarillo';
  return 'rojo';
}
/* =============================================
   FIN DE FUNCIÓN DE CLASIFICACIÓN
   ============================================= */

/* =============================================
   NAVEGACIÓN ENTRE PANTALLAS
   ============================================= */
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const target = document.getElementById(id);
  target.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* =============================================
   QUIZ — renderizar pregunta actual
   ============================================= */
function renderQuestion() {
  const q = questions[state.currentQuestion];
  const total = questions.length;
  const current = state.currentQuestion + 1;

  // Encabezado
  document.getElementById('quiz-category').textContent = q.category;
  document.getElementById('quiz-counter').textContent = `Pregunta ${current} de ${total}`;
  document.getElementById('progress-fill').style.width = `${((current - 1) / total) * 100}%`;

  // Texto
  document.getElementById('quiz-question').textContent = q.text;

  // Opciones
  const container = document.getElementById('quiz-options');
  container.innerHTML = '';
  delete container.dataset.selected;

  const letters = ['A', 'B', 'C'];
  q.options.forEach((opt, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'option-btn';
    btn.dataset.value = opt.value;
    btn.innerHTML = `
      <span class="option-letter" aria-hidden="true">${letters[i]}</span>
      <span>${opt.label}</span>
    `;
    btn.addEventListener('click', () => selectOption(btn, opt.value));
    container.appendChild(btn);
  });

  // Botón siguiente
  const btnNext = document.getElementById('btn-next');
  btnNext.disabled = true;
  btnNext.textContent = current < total ? 'Siguiente →' : 'Ver mi resultado';
}

/* --- Marcar opción seleccionada --- */
function selectOption(selected, value) {
  document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
  selected.classList.add('selected');
  document.getElementById('quiz-options').dataset.selected = value;
  document.getElementById('btn-next').disabled = false;
}

/* --- Avanzar a la siguiente pregunta --- */
function nextQuestion() {
  const raw = document.getElementById('quiz-options').dataset.selected;
  if (!raw) return;

  state.answers[state.currentQuestion] = parseInt(raw, 10);

  if (state.currentQuestion < questions.length - 1) {
    state.currentQuestion++;
    renderQuestion();
  } else {
    showResults();
  }
}

/* =============================================
   RESULTADOS
   ============================================= */
function showResults() {
  const nivel = clasificar(state.answers);

  const config = {
    verde: {
      icon: '🟢',
      title: 'Estás bien — ¡Sigue así!',
      subtitle: `${state.name}, tus respuestas muestran una buena adaptación. Aprovecha estos espacios para seguir creciendo en la universidad.`,
    },
    amarillo: {
      icon: '🟡',
      title: 'Riesgo moderado — Hay apoyo para ti',
      subtitle: `${state.name}, hemos notado algunas señales que merecen atención. No estás solo/a — en Unisabana hay personas listas para acompañarte.`,
    },
    rojo: {
      icon: '🔴',
      title: 'Riesgo alto — Busca apoyo hoy',
      subtitle: `${state.name}, tus respuestas indican que estás pasando un momento muy difícil. Por favor comunícate hoy mismo con Bienestar Unisabana — están ahí para ayudarte.`,
    },
  };

  const cfg = config[nivel];

  // Aplicar nivel al encabezado de resultados
  const header = document.getElementById('result-header');
  header.className = `result-header ${nivel}`;
  document.getElementById('result-icon').textContent = cfg.icon;
  document.getElementById('result-title').textContent = cfg.title;
  document.getElementById('result-subtitle').textContent = cfg.subtitle;

  // Renderizar lista de recursos
  const list = document.getElementById('resources-list');
  list.innerHTML = '';
  resources[nivel].forEach(r => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="resource-icon" aria-hidden="true">${r.icon}</span><span>${r.text}</span>`;
    list.appendChild(li);
  });

  showScreen('screen-results');
}

/* =============================================
   REINICIAR APLICACIÓN
   ============================================= */
function resetApp() {
  state.name = '';
  state.semester = '';
  state.answers = [];
  state.currentQuestion = 0;
  document.getElementById('student-name').value = '';
  document.getElementById('student-semester').value = '';
  document.getElementById('welcome-error').classList.add('hidden');
  showScreen('screen-welcome');
}

/* =============================================
   MODAL DE PÁNICO
   ============================================= */
function openPanicModal() {
  document.getElementById('panic-modal').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  document.getElementById('btn-modal-close').focus();
}

function closePanicModal() {
  document.getElementById('panic-modal').classList.add('hidden');
  document.body.style.overflow = '';
  // Limpiar formulario al cerrar
  document.getElementById('alert-message').value = '';
  document.getElementById('alert-confirmation').classList.add('hidden');
  document.getElementById('btn-send-alert').style.display = '';
}

/* --- Envío simulado de alerta --- */
function sendAlert() {
  const message = document.getElementById('alert-message').value.trim();
  if (!message) {
    document.getElementById('alert-message').focus();
    return;
  }
  // Simular envío: ocultar botón y mostrar confirmación
  document.getElementById('btn-send-alert').style.display = 'none';
  document.getElementById('alert-confirmation').classList.remove('hidden');
}

/* =============================================
   INICIALIZACIÓN
   ============================================= */
document.addEventListener('DOMContentLoaded', () => {

  /* --- Bienvenida --- */
  document.getElementById('form-welcome').addEventListener('submit', e => {
    e.preventDefault();
    const name = document.getElementById('student-name').value.trim();
    const semester = document.getElementById('student-semester').value;
    const errorEl = document.getElementById('welcome-error');

    if (!name || !semester) {
      errorEl.classList.remove('hidden');
      return;
    }
    errorEl.classList.add('hidden');

    state.name = name;
    state.semester = semester;
    state.currentQuestion = 0;
    state.answers = [];

    showScreen('screen-quiz');
    renderQuestion();
  });

  /* --- Quiz --- */
  document.getElementById('btn-next').addEventListener('click', nextQuestion);

  /* --- Resultado --- */
  document.getElementById('btn-restart').addEventListener('click', resetApp);

  /* --- Botón de pánico --- */
  document.getElementById('panic-btn').addEventListener('click', openPanicModal);

  /* --- Modal: cerrar con botón X --- */
  document.getElementById('btn-modal-close').addEventListener('click', closePanicModal);

  /* --- Modal: cerrar al hacer clic en el fondo --- */
  document.getElementById('panic-modal').addEventListener('click', e => {
    if (e.target === document.getElementById('panic-modal')) closePanicModal();
  });

  /* --- Modal: cerrar con tecla Escape --- */
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !document.getElementById('panic-modal').classList.contains('hidden')) {
      closePanicModal();
    }
  });

  /* --- Envío de alerta --- */
  document.getElementById('btn-send-alert').addEventListener('click', sendAlert);
});
