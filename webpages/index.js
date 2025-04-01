const map = L.map('map', {
    center: [50.8050316386387, -1.070178225201612],
    zoom: 13,
    dragging: false,
    zoomControl: false,
});

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

// adding image overlay
let imageUrl = '/guassion_plume.png';
const imageBounds = [[50.8050316386387, -1.070178225201612], [50.8050316386387, -1.070178225201612]];
L.imageOverlay(imageUrl, imageBounds).addTo(map);


