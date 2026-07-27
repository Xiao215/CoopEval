/* Interactive "Mechanisms in Motion" illustration.
   Vanilla JS + inline SVG; one scripted scene per mechanism,
   scrubbed by the timeline slider or auto-played. */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  var C = {
    coop: '#4f7748',
    coopSoft: '#e7f2e1',
    coopText: '#3e6338',
    def: '#ad4f5b',
    defSoft: '#fae5e8',
    defText: '#8b3b47',
    ink: '#3b4657',
    muted: '#8a93a3',
    past: '#c9d2de',
    line: '#dbe1ea'
  };

  var MECH_COLOR = {
    repetition: '#2f5f8f',
    reputation: '#a36f2e',
    mediation: '#ad4f5b',
    contracting: '#4f7748'
  };

  function make(parent, name, attrs) {
    var n = document.createElementNS(NS, name);
    if (attrs) {
      for (var k in attrs) n.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function text(parent, x, y, str, attrs) {
    var t = make(parent, 'text', Object.assign({
      x: x, y: y, 'text-anchor': 'middle', dy: '0.35em'
    }, attrs || {}));
    t.textContent = str;
    return t;
  }

  function agentNode(parent, x, y, r, label) {
    var g = make(parent, 'g', { transform: 'translate(' + x + ',' + y + ')' });
    make(g, 'circle', { r: r, fill: '#ffffff', stroke: C.ink, 'stroke-width': 2 });
    text(g, 0, 0, label, {
      'font-size': r >= 30 ? 17 : 13,
      'font-weight': 700,
      fill: C.ink
    });
    return { g: g, x: x, y: y, r: r };
  }

  /* Action chip shown above an agent. Full chips say Cooperate/Defect,
     compact chips show just C/D. */
  function chip(parent, x, y, compact) {
    var g = make(parent, 'g', { transform: 'translate(' + x + ',' + y + ')', opacity: 0 });
    var w = compact ? 26 : 96;
    var h = compact ? 20 : 26;
    var rect = make(g, 'rect', {
      x: -w / 2, y: -h / 2, width: w, height: h, rx: h / 2, 'stroke-width': 1.5
    });
    var txt = text(g, 0, 0.5, '', { 'font-size': compact ? 11.5 : 13, 'font-weight': 700 });
    return {
      set: function (action) {
        if (!action) { g.setAttribute('opacity', 0); return; }
        g.setAttribute('opacity', 1);
        var coop = action === 'C';
        rect.setAttribute('fill', coop ? C.coopSoft : C.defSoft);
        rect.setAttribute('stroke', coop ? C.coop : C.def);
        txt.setAttribute('fill', coop ? C.coopText : C.defText);
        txt.textContent = compact ? action : (coop ? 'Cooperate' : 'Defect');
      }
    };
  }

  /* Line between two agents, trimmed so it doesn't enter the circles. */
  function link(parent, a, b, attrs) {
    var dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
    var ux = dx / d, uy = dy / d;
    return make(parent, 'line', Object.assign({
      x1: a.x + ux * (a.r + 5), y1: a.y + uy * (a.r + 5),
      x2: b.x - ux * (b.r + 5), y2: b.y - uy * (b.r + 5)
    }, attrs));
  }

  /* ---------------- Repetition ---------------- */
  function buildRepetition(svg) {
    var root = make(svg, 'g');
    var actions = [['C', 'C'], ['C', 'C'], ['C', 'D'], ['D', 'C'], ['C', 'C'], ['C', 'C']];
    var steps = actions.length;

    var a = agentNode(root, 195, 128, 34, 'A');
    var b = agentNode(root, 525, 128, 34, 'B');
    var line = link(root, a, b, { stroke: C.line, 'stroke-width': 2 });
    var roundLabel = text(root, 360, 106, '', {
      'font-size': 13, 'font-weight': 600, fill: C.muted
    });
    text(root, 195, 180, 'same co-player', { 'font-size': 12.5, fill: C.muted });
    text(root, 525, 180, 'every round', { 'font-size': 12.5, fill: C.muted });

    var chipA = chip(root, 195, 74);
    var chipB = chip(root, 525, 74);

    // Shared-history strip: one column per round, A's action on top of B's.
    text(root, 360, 218, 'SHARED HISTORY', {
      'font-size': 11, 'font-weight': 700, fill: C.muted, 'letter-spacing': '0.12em'
    });
    var slotX0 = 360 - (steps * 38) / 2 + 19;
    text(root, slotX0 - 40, 244, 'A', { 'font-size': 11, 'font-weight': 700, fill: C.muted });
    text(root, slotX0 - 40, 264, 'B', { 'font-size': 11, 'font-weight': 700, fill: C.muted });
    var slots = [];
    for (var i = 0; i < steps; i++) {
      var x = slotX0 + i * 38;
      slots.push({
        a: make(root, 'rect', { x: x - 8, y: 236, width: 16, height: 16, rx: 4 }),
        b: make(root, 'rect', { x: x - 8, y: 256, width: 16, height: 16, rx: 4 })
      });
    }

    function paintSlot(rect, action, on) {
      if (!on) {
        rect.setAttribute('fill', '#f4f6f9');
        rect.setAttribute('stroke', C.line);
        rect.setAttribute('stroke-dasharray', '3 2');
        return;
      }
      rect.removeAttribute('stroke-dasharray');
      var coop = action === 'C';
      rect.setAttribute('fill', coop ? C.coopSoft : C.defSoft);
      rect.setAttribute('stroke', coop ? C.coop : C.def);
    }

    return {
      root: root,
      steps: steps,
      captions: [
        'Round 1 — the same two agents meet in the dilemma, and both cooperate.',
        'Round 2 — the game repeats: today’s action shapes tomorrow’s game.',
        'Round 3 — “B” defects, grabbing a short-term payoff.',
        'Round 4 — “A” remembers the history and retaliates.',
        'Round 5 — “B” returns to cooperating, and “A” forgives.',
        'Round 6 — the shadow of the future keeps cooperation stable.'
      ],
      update: function (t) {
        var cur = actions[t - 1];
        chipA.set(cur[0]);
        chipB.set(cur[1]);
        roundLabel.textContent = 'Round ' + t + ' of ' + steps;
        var bothC = cur[0] === 'C' && cur[1] === 'C';
        line.setAttribute('stroke', bothC ? '#9dbb98' : '#d3a2aa');
        for (var i = 0; i < steps; i++) {
          paintSlot(slots[i].a, actions[i][0], i < t);
          paintSlot(slots[i].b, actions[i][1], i < t);
        }
      }
    };
  }

  /* ---------------- Reputation ---------------- */
  function buildReputation(svg) {
    var root = make(svg, 'g');
    // step: [i, j, action_i, action_j]
    var script = [
      [0, 1, 'C', 'C'],
      [2, 3, 'C', 'C'],
      [4, 5, 'C', 'D'],
      [1, 5, 'D', 'D'],
      [0, 2, 'C', 'C'],
      [3, 5, 'D', 'D']
    ];
    var steps = script.length;
    var cx = 360, cy = 146, R = 92;
    var names = ['A', 'B', 'C', 'D', 'E', 'F'];

    var edgeLayer = make(root, 'g');
    var agents = [], chips = [];
    for (var k = 0; k < 6; k++) {
      var ang = (-90 + k * 60) * Math.PI / 180;
      var x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
      agents.push(agentNode(root, x, y, 21, names[k]));
      // Chips sit radially outside the ring so they never cross the edges.
      chips.push(chip(root, x + 37 * Math.cos(ang), y + 37 * Math.sin(ang), true));
    }

    var edges = script.map(function (s) {
      return link(edgeLayer, agents[s[0]], agents[s[1]], { opacity: 0 });
    });

    // Public record: a dot per past action, shown under each agent.
    var records = agents.map(function () { return []; });
    script.forEach(function (s, idx) {
      records[s[0]].push({ step: idx + 1, action: s[2] });
      records[s[1]].push({ step: idx + 1, action: s[3] });
    });
    var dots = records.map(function (rec, k) {
      var ag = agents[k];
      return rec.map(function (r, i) {
        var dx = (i - (rec.length - 1) / 2) * 9;
        return make(root, 'circle', {
          cx: ag.x + dx, cy: ag.y + 11.5, r: 2.9,
          fill: r.action === 'C' ? C.coop : C.def, opacity: 0
        });
      });
    });
    text(root, 155, 285, 'dots = each agent’s public record', {
      'font-size': 12.5, fill: C.muted
    });

    return {
      root: root,
      steps: steps,
      captions: [
        'Round 1 — “A” and “B” are matched. Both cooperate, and the actions go on their public records.',
        'Round 2 — new pairs form every round: “C” and “D” also cooperate.',
        'Round 3 — “F” defects on “E”. The defection is written into the public record of “F”.',
        'Round 4 — “B” is matched with “F”, reads its record, and defects to protect itself.',
        'Round 5 — agents with clean records keep finding cooperation with new partners.',
        'Round 6 — the reputation of “F” follows it into every new match.'
      ],
      update: function (t) {
        for (var s = 0; s < steps; s++) {
          var e = edges[s];
          if (s + 1 > t) { e.setAttribute('opacity', 0); continue; }
          var current = (s + 1 === t);
          e.setAttribute('opacity', current ? 1 : 0.55);
          e.setAttribute('stroke', current ? MECH_COLOR.reputation : C.past);
          e.setAttribute('stroke-width', current ? 3 : 2);
        }
        var cur = script[t - 1];
        chips.forEach(function (ch, k) {
          if (k === cur[0]) ch.set(cur[2]);
          else if (k === cur[1]) ch.set(cur[3]);
          else ch.set(null);
        });
        dots.forEach(function (list, k) {
          list.forEach(function (dot, i) {
            dot.setAttribute('opacity', records[k][i].step <= t ? 1 : 0);
          });
        });
      }
    };
  }

  /* ---------------- Mediation ---------------- */
  function buildMediation(svg) {
    var root = make(svg, 'g');
    var steps = 6;
    var cx = 360, cy = 146, R = 96;
    var names = ['A', 'B', 'C', 'D', 'E'];
    var delegateAt = [2, 3, 3, 4, Infinity]; // step each agent delegates; E never does

    var edgeLayer = make(root, 'g');
    var med = { x: cx, y: cy, r: 26 };
    var agents = [], chips = [];
    for (var k = 0; k < 5; k++) {
      var ang = (-90 + k * 72) * Math.PI / 180;
      var x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
      agents.push(agentNode(root, x, y, 21, names[k]));
      chips.push(chip(root, x + 37 * Math.cos(ang), y + 37 * Math.sin(ang), true));
    }

    var medG = make(root, 'g', { transform: 'translate(' + cx + ',' + cy + ')' });
    var medCircle = make(medG, 'circle', {
      r: med.r, fill: C.defSoft, stroke: MECH_COLOR.mediation, 'stroke-width': 2
    });
    text(medG, 0, 0, 'M', { 'font-size': 16, 'font-weight': 800, fill: C.defText });
    text(root, cx, cy + med.r + 15, 'mediator', { 'font-size': 12.5, fill: C.muted });

    var edges = agents.map(function (ag) {
      return link(edgeLayer, { x: cx, y: cy, r: med.r }, ag, {
        stroke: MECH_COLOR.mediation, opacity: 0
      });
    });
    var eNote = text(root, agents[4].x - 21 - 8, agents[4].y + 30, 'keeps its own move', {
      'font-size': 12.5, fill: C.muted, opacity: 0
    });

    return {
      root: root,
      steps: steps,
      captions: [
        'A mediator “M” offers to act on behalf of any agent that delegates its move.',
        '“A” delegates its decision to the mediator.',
        '“B” and “C” delegate too — the mediator’s commitment grows.',
        '“D” joins as well; “E” prefers to keep its own move.',
        '“M” plays Cooperate on behalf of every delegator, all at once.',
        'With most of the group committed through “M”, cooperating becomes the best response even for “E”.'
      ],
      update: function (t) {
        medCircle.classList.toggle('tv-pulse', t === 1);
        for (var k = 0; k < 5; k++) {
          var joined = t >= delegateAt[k];
          var isNew = t === delegateAt[k];
          edges[k].setAttribute('opacity', joined ? (isNew ? 1 : 0.7) : 0);
          edges[k].setAttribute('stroke-width', isNew ? 3 : 2);
          edges[k].setAttribute('stroke-dasharray', isNew ? 'none' : 'none');
          var action = null;
          if (t >= 5 && delegateAt[k] < Infinity) action = 'C';
          if (t >= 6) action = 'C';
          chips[k].set(action);
        }
        eNote.setAttribute('opacity', t >= 4 && t < 6 ? 1 : 0);
      }
    };
  }

  /* ---------------- Contracting ---------------- */
  function buildContracting(svg) {
    var root = make(svg, 'g');
    var steps = 6;

    var a = agentNode(root, 195, 132, 34, 'A');
    var b = agentNode(root, 525, 132, 34, 'B');
    var chipA = chip(root, 195, 78);
    var chipB = chip(root, 525, 78);

    // Contract document between the players.
    var doc = make(root, 'g', { transform: 'translate(360,124)' });
    var docRect = make(doc, 'rect', {
      x: -23, y: -30, width: 46, height: 60, rx: 6,
      fill: '#ffffff', stroke: C.ink, 'stroke-width': 2
    });
    [-14, -4, 6].forEach(function (y) {
      make(doc, 'line', {
        x1: -13, y1: y, x2: 13, y2: y, stroke: C.past, 'stroke-width': 2.5,
        'stroke-linecap': 'round'
      });
    });
    make(doc, 'line', {
      x1: -13, y1: 18, x2: 3, y2: 18, stroke: MECH_COLOR.contracting,
      'stroke-width': 2.5, 'stroke-linecap': 'round'
    });
    var check = make(doc, 'g', { transform: 'translate(18,-25)', opacity: 0 });
    make(check, 'circle', { r: 9, fill: C.coop });
    make(check, 'path', {
      d: 'M -4 0 L -1 3 L 4 -3', stroke: '#ffffff', 'stroke-width': 2,
      fill: 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
    });
    var clause = text(root, 360, 176, 'clause: a defector pays the other player', {
      'font-size': 13.5, 'font-weight': 600, fill: C.coopText, opacity: 0
    });

    // Transfer payment arc from B to A.
    var transfer = make(root, 'g', { opacity: 0 });
    make(transfer, 'path', {
      d: 'M 512 194 Q 360 262 212 196',
      fill: 'none', stroke: MECH_COLOR.contracting, 'stroke-width': 2.5,
      'marker-end': 'url(#tv-arrow)'
    });
    var coin = make(transfer, 'g', { transform: 'translate(360,228)' });
    make(coin, 'circle', { r: 13, fill: '#fff7d4', stroke: '#a36f2e', 'stroke-width': 2 });
    text(coin, 0, 0.5, '$', { 'font-size': 13, 'font-weight': 800, fill: '#805421' });
    var transferLabel = text(root, 360, 264, '“B” pays the agreed transfer to “A”', {
      'font-size': 12.5, fill: C.muted, opacity: 0
    });

    return {
      root: root,
      steps: steps,
      captions: [
        '“A” proposes a contract: outcome-conditional payments between the two players.',
        '“B” signs — the terms are now binding for both.',
        'The key clause: whoever defects pays a transfer to the other player.',
        '“B” tests the contract and defects anyway…',
        '…but the transfer is automatic: defection no longer pays.',
        'With incentives realigned, both players settle into cooperation.'
      ],
      update: function (t) {
        docRect.setAttribute('stroke-dasharray', t === 1 ? '5 4' : 'none');
        doc.setAttribute('opacity', t === 1 ? 0.75 : 1);
        check.setAttribute('opacity', t >= 2 ? 1 : 0);
        clause.setAttribute('opacity', t >= 3 ? 1 : 0);
        if (t === 4 || t === 5) { chipA.set('C'); chipB.set('D'); }
        else if (t === 6) { chipA.set('C'); chipB.set('C'); }
        else { chipA.set(null); chipB.set(null); }
        transfer.setAttribute('opacity', t === 5 ? 1 : 0);
        transferLabel.setAttribute('opacity', t === 5 ? 1 : 0);
      }
    };
  }

  /* ---------------- Controller ---------------- */
  function init() {
    var widget = document.querySelector('.tournament-widget');
    if (!widget) return;

    var svg = widget.querySelector('svg');
    var slider = widget.querySelector('.tournament-slider');
    var playBtn = widget.querySelector('.tournament-play');
    var stepLabel = widget.querySelector('.tournament-step');
    var caption = widget.querySelector('.tournament-caption');
    var tabs = Array.prototype.slice.call(widget.querySelectorAll('.tournament-tab'));

    var defs = make(svg, 'defs');
    var marker = make(defs, 'marker', {
      id: 'tv-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5,
      markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse'
    });
    make(marker, 'path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: MECH_COLOR.contracting });

    var scenes = {
      repetition: buildRepetition(svg),
      reputation: buildReputation(svg),
      mediation: buildMediation(svg),
      contracting: buildContracting(svg)
    };
    Object.keys(scenes).forEach(function (k) {
      scenes[k].root.setAttribute('display', 'none');
    });

    var mech = 'repetition';
    var t = 1;
    var timer = null;
    var reduceMotion = window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function render() {
      var scene = scenes[mech];
      scene.update(t);
      caption.textContent = scene.captions[t - 1];
      slider.value = t;
      stepLabel.textContent = t + ' / ' + scene.steps;
    }

    function setPlaying(on) {
      if (on && !timer) {
        timer = setInterval(function () {
          t = t >= scenes[mech].steps ? 1 : t + 1;
          render();
        }, 1800);
      } else if (!on && timer) {
        clearInterval(timer);
        timer = null;
      }
      playBtn.classList.toggle('is-playing', !!timer);
      playBtn.setAttribute('aria-label', timer ? 'Pause' : 'Play');
    }

    function setMech(k) {
      scenes[mech].root.setAttribute('display', 'none');
      mech = k;
      t = 1;
      scenes[mech].root.removeAttribute('display');
      slider.max = scenes[mech].steps;
      slider.style.accentColor = MECH_COLOR[mech];
      playBtn.style.background = MECH_COLOR[mech];
      tabs.forEach(function (tab) {
        var active = tab.dataset.mech === k;
        tab.classList.toggle('is-active', active);
        tab.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      render();
    }

    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () { setMech(tab.dataset.mech); });
    });

    playBtn.addEventListener('click', function () { setPlaying(!timer); });

    slider.addEventListener('input', function () {
      setPlaying(false);
      t = parseInt(slider.value, 10) || 1;
      render();
    });

    // Auto-play once the widget scrolls into view; pause when it leaves.
    if ('IntersectionObserver' in window && !reduceMotion) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) { setPlaying(entry.isIntersecting); });
      }, { threshold: 0.35 }).observe(widget);
    }

    // Optional deep link, e.g. ?mech=reputation&step=3
    var params = new URLSearchParams(window.location.search);
    var startMech = params.get('mech');
    setMech(scenes[startMech] ? startMech : 'repetition');
    var startStep = parseInt(params.get('step'), 10);
    if (startStep >= 1 && startStep <= scenes[mech].steps) {
      t = startStep;
      render();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
