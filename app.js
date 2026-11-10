const mapContainer = document.getElementById('career-map');
const filterSelect = document.getElementById('career-level-filter');
const legendContainer = document.getElementById('career-level-legend');

const careerLevelPalette = {
  'Entry-Level': '#2f7f79',
  'Experienced (non-manager)': '#7c8f5f',
  'Manager': '#d08c4f',
  'Senior Manager': '#b45f6a',
  'Executive': '#5b6bb0',
  'Senior Executive': '#6d4e8a',
  'Unknown': '#8a948d'
};

let allPoints = [];

function getLevelColor(level) {
  return careerLevelPalette[level] || careerLevelPalette.Unknown;
}

function renderLegend(levels) {
  if (!legendContainer) return;

  const items = levels.map((level) => {
    const color = getLevelColor(level);
    return `
      <div class="legend-item">
        <span class="legend-swatch" style="background:${color};"></span>
        <span>${level}</span>
      </div>
    `;
  }).join('');

  legendContainer.innerHTML = items;
}

function renderFilterOptions(levels) {
  if (!filterSelect) return;

  const options = ['All', ...levels].map((level) => {
    const label = level === 'All' ? 'All levels' : level;
    return `<option value="${level}">${label}</option>`;
  }).join('');

  filterSelect.innerHTML = options;
}

function getVisiblePoints(selectedLevel) {
  if (!selectedLevel || selectedLevel === 'All') {
    return allPoints;
  }

  return allPoints.filter((point) => point.careerLevel === selectedLevel);
}

function drawMap(points) {
  if (!mapContainer) return;

  const trace = {
    type: 'scatter3d',
    mode: 'markers',
    x: points.map((point) => Number(point.x)),
    y: points.map((point) => Number(point.y)),
    z: points.map((point) => Number(point.z)),
    text: points.map((point) => point.businessTitle || 'Job title'),
    customdata: points.map((point) => ({
      level: point.careerLevel || 'Unknown',
      agency: point.agency || 'Unknown agency'
    })),
    hovertemplate: '%{text}<br>Career level: %{customdata.level}<br>Agency: %{customdata.agency}<extra></extra>',
    marker: {
      size: 5,
      color: points.map((point) => getLevelColor(point.careerLevel)),
      opacity: 0.9,
      line: {
        color: '#1d2a2a',
        width: 0.2
      }
    }
  };

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: '#f6f7f4',
    plot_bgcolor: '#f6f7f4',
    scene: {
      xaxis: { title: 'UMAP X', backgroundcolor: '#f7faf8', gridcolor: '#dce5df', showbackground: true },
      yaxis: { title: 'UMAP Y', backgroundcolor: '#f7faf8', gridcolor: '#dce5df', showbackground: true },
      zaxis: { title: 'UMAP Z', backgroundcolor: '#f7faf8', gridcolor: '#dce5df', showbackground: true },
      camera: { eye: { x: 1.7, y: 1.7, z: 1.3 } },
      bgcolor: '#f7faf8'
    },
    hoverlabel: {
      font: { family: 'Segoe UI, sans-serif', color: '#1d2a2a' }
    }
  };

  Plotly.react('career-map', [trace], layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d']
  });
}

async function loadMap() {
  try {
    if (!mapContainer) {
      throw new Error('The career map container could not be found.');
    }

    const response = await fetch('./data/umap_points.json');
    if (!response.ok) {
      throw new Error(`Unable to load UMAP data (${response.status}).`);
    }

    const points = await response.json();
    if (!Array.isArray(points) || points.length === 0) {
      throw new Error('No map points were returned from the UMAP dataset.');
    }

    allPoints = points;

    const levels = [...new Set(points.map((point) => point.careerLevel || 'Unknown'))].sort((a, b) => a.localeCompare(b));
    renderLegend(levels);
    renderFilterOptions(levels);

    const updateSelectedLevel = () => {
      const value = filterSelect ? filterSelect.value : 'All';
      drawMap(getVisiblePoints(value));
    };

    if (filterSelect) {
      filterSelect.addEventListener('change', updateSelectedLevel);
    }

    updateSelectedLevel();

    window.addEventListener('resize', () => {
      Plotly.Plots.resize(mapContainer);
    });
  } catch (error) {
    mapContainer.innerHTML = `
      <div class="map-error">
        <div>
          <strong>UMAP map unavailable.</strong><br>
          ${error.message}<br>
          Run a local server and generate the file from the project data to display the 3D map.
        </div>
      </div>
    `;
  }
}

loadMap();
