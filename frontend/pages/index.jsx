import React, { useState, useEffect } from 'react'
import { fetchHealth, fetchEngines, predictRUL, getMaintenanceRecommendation, chat } from '../lib/api'

// --- TERMINAL COMPONENTS ---

const TerminalHeader = ({ healthStatus }) => (
  <div style={{
    background: 'rgba(5, 10, 20, 0.95)',
    borderBottom: '2px solid #00f3ff',
    padding: '15px 30px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    boxShadow: '0 0 30px rgba(0, 243, 255, 0.2)'
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
      <div style={{
        width: '50px', height: '50px',
        border: '2px solid var(--neon-cyan)',
        borderRadius: '50%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 0 15px var(--neon-cyan)',
        background: 'rgba(0, 243, 255, 0.1)'
      }}>
        <span style={{ color: 'var(--neon-cyan)', fontSize: '1.8rem' }}>⚛</span>
      </div>
      <div>
        <h1 className="text-cyan" style={{ fontSize: '2rem', margin: 0, letterSpacing: '3px' }}>AERO-SENSE <span style={{ color: '#fff', fontSize: '1rem' }}>DIAGNOSTICS</span></h1>
        <div style={{ fontFamily: 'Rajdhani', fontSize: '0.9rem', color: '#aaa', letterSpacing: '1px' }}>
          SYSTEM STATUS: {healthStatus?.status === 'healthy' ? (
            <span className="text-green">● ONLINE</span>
          ) : healthStatus?.status === 'degraded' ? (
            <span className="text-orange">● DEGRADED</span>
          ) : (
            <span className="text-red">● OFFLINE</span>
          )} // V.{healthStatus?.version || '1.0.0'}
        </div>
      </div>
    </div>
    <div style={{ display: 'flex', gap: '30px' }}>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: '0.7rem', color: '#888' }}>DATASET</div>
        <div className="text-cyan">{healthStatus?.dataset || 'FD001'}</div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: '0.7rem', color: '#888' }}>MODELS</div>
        <div className={healthStatus?.models_loaded ? 'text-green' : 'text-red'} style={{ fontWeight: 'bold' }}>
          {healthStatus?.models_loaded ? 'LOADED' : 'NOT LOADED'}
        </div>
      </div>
    </div>
  </div>
)

const MetricCard = ({ label, value, unit, status = 'nominal', description }) => {
  const colorClass = status === 'danger' ? 'text-red' : status === 'warning' ? 'text-orange' : 'text-cyan';
  const borderClass = status === 'danger' ? '4px solid #ff003c' : status === 'warning' ? '4px solid #ffaa00' : '4px solid #00f3ff';

  return (
    <div className="metric-card shadow-cyan" style={{
      padding: '12px 15px',
      borderRadius: '10px',
      borderLeft: borderClass,
      background: 'rgba(0, 243, 255, 0.05)',
      position: 'relative',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      minHeight: '100px'
    }} title={description}>
      <div style={{ position: 'absolute', top: 5, right: 5, padding: '5px', opacity: 0.1, fontSize: '2rem', color: '#fff' }}>
        {status === 'danger' ? '⚠' : status === 'warning' ? '⚡' : '∿'}
      </div>
      <div>
        <div style={{ color: 'var(--neon-cyan)', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '2px', fontWeight: 'bold' }}>{label}</div>
        <div style={{ fontSize: '0.6rem', color: 'rgba(255,255,255,0.4)', marginBottom: '5px', fontStyle: 'italic' }}>{description}</div>
      </div>
      <div style={{ color: '#fff', fontSize: '1.8rem', fontFamily: 'Orbitron', lineHeight: 1.2 }}>
        {value} <span style={{ fontSize: '0.8rem', verticalAlign: 'middle' }} className={colorClass}>{unit}</span>
      </div>
      <div className={colorClass} style={{ fontSize: '0.65rem', marginTop: '5px', fontWeight: 'bold', textTransform: 'uppercase' }}>
        {status}
      </div>
    </div>
  )
}

const PredictionCard = ({ prediction }) => {
  if (!prediction) return null;

  const getStateColor = (state) => {
    if (state === 'critical') return 'text-red';
    if (state === 'degrading') return 'text-orange';
    return 'text-green';
  };

  const getUrgencyBorder = (rul) => {
    if (rul <= 50) return '3px solid #ff003c';
    if (rul <= 125) return '3px solid #ffaa00';
    return '3px solid #0aff00';
  };

  return (
    <div className="glow-box" style={{
      padding: '20px',
      borderRadius: '10px',
      borderLeft: getUrgencyBorder(prediction.predicted_rul),
      marginBottom: '15px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h3 className="text-cyan" style={{ margin: 0 }}>ENGINE UNIT #{prediction.unit_id}</h3>
        <span className={getStateColor(prediction.state)} style={{
          padding: '5px 15px',
          border: '1px solid currentColor',
          borderRadius: '4px',
          fontFamily: 'Orbitron',
          fontSize: '0.8rem'
        }}>
          {prediction.state?.toUpperCase()}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div>
          <div style={{ color: '#888', fontSize: '0.7rem', marginBottom: '5px' }}>PREDICTED RUL</div>
          <div style={{ fontSize: '2.5rem', fontFamily: 'Orbitron', color: '#fff' }}>
            {Math.round(prediction.predicted_rul)}
            <span style={{ fontSize: '1rem', color: '#888' }}> cycles</span>
          </div>
        </div>
        <div>
          <div style={{ color: '#888', fontSize: '0.7rem', marginBottom: '5px' }}>CONFIDENCE</div>
          <div className={prediction.confidence === 'high' ? 'text-green' : prediction.confidence === 'medium' ? 'text-orange' : 'text-red'}
            style={{ fontSize: '1.2rem', fontFamily: 'Orbitron', textTransform: 'uppercase' }}>
            {prediction.confidence}
          </div>
        </div>
      </div>

      <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <div style={{ color: '#888', fontSize: '0.7rem', marginBottom: '8px' }}>STATE PROBABILITIES</div>
        <div style={{ display: 'flex', gap: '15px' }}>
          {Object.entries(prediction.state_probabilities || {}).map(([state, prob]) => (
            <div key={state} style={{ flex: 1, textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: '#666', textTransform: 'uppercase' }}>{state}</div>
              <div style={{
                height: '4px',
                background: 'rgba(255,255,255,0.1)',
                borderRadius: '2px',
                marginTop: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${prob * 100}%`,
                  height: '100%',
                  background: state === 'critical' ? '#ff003c' : state === 'degrading' ? '#ffaa00' : '#0aff00',
                  transition: 'width 0.5s'
                }}></div>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#fff', marginTop: '2px' }}>{(prob * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const TerminalOutput = ({ answer, loading, prediction }) => {
  if (loading) return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: '80px', height: '80px', border: '4px solid transparent', borderTop: '4px solid var(--neon-cyan)', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
      <div className="text-cyan" style={{ marginTop: '20px', letterSpacing: '2px', fontFamily: 'Orbitron' }}>RUNNING DIAGNOSTICS...</div>
    </div>
  )

  if (!answer && !prediction) return (
    <div style={{
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      opacity: 0.5,
      flexDirection: 'column',
      background: 'repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(0,243,255,0.05) 10px, rgba(0,243,255,0.05) 20px)'
    }}>
      <div style={{ fontSize: '4rem', color: 'var(--neon-cyan)', marginBottom: '20px', textShadow: '0 0 20px var(--neon-cyan)' }}>⚡</div>
      <div style={{ fontFamily: 'Orbitron', letterSpacing: '2px', color: '#fff' }}>SYSTEM READY</div>
      <div style={{ color: '#888', marginTop: '10px', textAlign: 'center' }}>
        Select an engine unit and run prediction<br />
        or use the command console for queries
      </div>
    </div>
  )

  return (
    <div className="glow-box" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, padding: '0', display: 'flex', flexDirection: 'column', overflow: 'hidden', borderRadius: '10px' }}>
      <div style={{ background: 'rgba(0,0,0,0.5)', padding: '15px', borderBottom: '1px solid rgba(0,243,255,0.2)', display: 'flex', justifyContent: 'space-between' }}>
        <div className="text-cyan" style={{ fontFamily: 'Orbitron' }}>DIAGNOSTIC REPORT</div>
        <div style={{ display: 'flex', gap: '10px' }}>
          {prediction && <span className="text-green">✓ PREDICTION COMPLETE</span>}
          {answer && <span className="text-cyan">✓ AI RESPONSE</span>}
        </div>
      </div>
      <div style={{
        padding: '20px',
        fontFamily: 'monospace',
        color: '#e0e0e0',
        lineHeight: '1.8',
        overflowY: 'auto',
        flex: 1,
        fontSize: '1rem'
      }}>
        {prediction && <PredictionCard prediction={prediction} />}
        {answer && (
          <div style={{ whiteSpace: 'pre-wrap', marginTop: prediction ? '20px' : 0 }}>
            {answer}
          </div>
        )}
      </div>
      <div style={{ padding: '10px', background: 'rgba(0,0,0,0.8)', color: '#666', fontSize: '0.8rem', textAlign: 'right', borderTop: '1px solid #333' }}>
        END OF REPORT // <span className="text-cyan">CONFIDENTIAL</span>
      </div>
    </div>
  )
}

export default function Home() {
  const [healthStatus, setHealthStatus] = useState(null)
  const [engines, setEngines] = useState([])
  const [selectedEngine, setSelectedEngine] = useState('')
  const [conversationId, setConversationId] = useState(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [maintenance, setMaintenance] = useState(null)
  const [loading, setLoading] = useState(false)
  const [predicting, setPredicting] = useState(false)


  // Fetch health status on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await fetchHealth()
        setHealthStatus(health)
      } catch (err) {
        setHealthStatus({ status: 'offline', models_loaded: false })
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  // Fetch engines when models are loaded
  useEffect(() => {
    const loadEngines = async () => {
      if (healthStatus?.models_loaded) {
        try {
          const data = await fetchEngines()
          setEngines(data.engines || [])
        } catch (err) {
          console.error('Failed to load engines:', err)
        }
      }
    }
    loadEngines()
  }, [healthStatus?.models_loaded])

  // Handle RUL prediction
  const handlePredict = async () => {
    if (!selectedEngine) return alert('Please select an engine unit')
    setPredicting(true)
    setPrediction(null)
    setMaintenance(null)

    try {
      const [predResult, maintResult] = await Promise.all([
        predictRUL(parseInt(selectedEngine)),
        getMaintenanceRecommendation(parseInt(selectedEngine))
      ])
      setPrediction(predResult)
      setMaintenance(maintResult)
    } catch (err) {
      alert(`Prediction failed: ${err.message}`)
    }
    setPredicting(false)
  }

  // Handle chat command
  const handleCommand = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    if (!healthStatus?.models_loaded) return alert('ERROR: MODELS NOT LOADED')

    setLoading(true)
    setAnswer('')

    try {
      const result = await chat(question, conversationId)
      setAnswer(result.response)
      setConversationId(result.conversation_id)
    } catch (err) {
      setAnswer(`>> CRITICAL FAILURE: ${err.message}`)
    }
    setLoading(false)
    setQuestion('')
  }



  // Get metrics from prediction for telemetry display
  const getRULStatus = () => {
    if (!prediction) return { value: '--', status: 'nominal' }
    const rul = prediction.predicted_rul
    return {
      value: Math.round(rul),
      status: rul <= 50 ? 'danger' : rul <= 125 ? 'warning' : 'nominal'
    }
  }

  const getStateStatus = () => {
    if (!prediction) return { value: '--', status: 'nominal' }
    const state = prediction.state
    return {
      value: state?.charAt(0).toUpperCase() + state?.slice(1),
      status: state === 'critical' ? 'danger' : state === 'degrading' ? 'warning' : 'nominal'
    }
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#0f172a' }}>
      <TerminalHeader healthStatus={healthStatus} />

      <div style={{ flex: 1, display: 'flex', padding: '30px', gap: '30px', overflow: 'hidden' }}>

        {/* LEFT PANEL: CONTROLS */}
        <div className="glow-box" style={{
          width: '380px',
          display: 'flex',
          flexDirection: 'column',
          gap: '18px',
          padding: '18px',
          borderRadius: '10px'
        }}>
          <h3 className="text-cyan" style={{ borderBottom: '1px solid rgba(0,243,255,0.3)', paddingBottom: '10px', margin: 0 }}>CONTROL PANEL</h3>

          {/* ENGINE SELECTION */}
          <div>
            <label style={{ display: 'block', fontSize: '0.7rem', color: '#aaa', marginBottom: '8px', letterSpacing: '1px' }}>SELECT ENGINE UNIT</label>
            <select
              value={selectedEngine}
              onChange={e => setSelectedEngine(e.target.value)}
              className="border-cyan"
              style={{
                background: 'rgba(0,0,0,0.5)',
                color: '#fff',
                padding: '12px',
                width: '100%',
                fontFamily: 'monospace',
                borderRadius: '4px',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value="">-- Select Unit --</option>
              {engines.map(eng => (
                <option key={eng.unit_id} value={eng.unit_id}>
                  Unit #{eng.unit_id} (Cycle: {eng.max_cycle})
                </option>
              ))}
            </select>
          </div>

          {/* PREDICT BUTTON */}
          <button
            className="system-btn shadow-cyan"
            onClick={handlePredict}
            disabled={!selectedEngine || predicting || !healthStatus?.models_loaded}
            style={{
              width: '100%',
              padding: '18px',
              fontSize: '1rem',
              fontWeight: 'bold',
              letterSpacing: '2px',
              background: 'rgba(0, 243, 255, 0.1)',
              border: '2px solid var(--neon-cyan)',
              borderRadius: '8px',
              opacity: (!selectedEngine || !healthStatus?.models_loaded) ? 0.3 : 1,
              transition: 'all 0.3s'
            }}
          >
            {predicting ? '⚡ ANALYZING SENSORS...' : 'PREDICT RUL'}
          </button>



          {/* MAINTENANCE ALERT */}
          {maintenance && (
            <div style={{
              padding: '15px',
              borderRadius: '8px',
              background: maintenance.urgency === 'CRITICAL' ? 'rgba(255,0,60,0.2)' :
                maintenance.urgency === 'MEDIUM' ? 'rgba(255,170,0,0.2)' : 'rgba(10,255,0,0.2)',
              border: `1px solid ${maintenance.urgency === 'CRITICAL' ? '#ff003c' :
                maintenance.urgency === 'MEDIUM' ? '#ffaa00' : '#0aff00'}`
            }}>
              <div style={{
                fontSize: '0.7rem',
                color: '#888',
                marginBottom: '8px',
                display: 'flex',
                justifyContent: 'space-between'
              }}>
                <span>MAINTENANCE ALERT</span>
                <span className={maintenance.urgency === 'CRITICAL' ? 'text-red' :
                  maintenance.urgency === 'MEDIUM' ? 'text-orange' : 'text-green'}>
                  {maintenance.urgency}
                </span>
              </div>
              <div style={{ fontSize: '0.85rem', color: '#fff', lineHeight: '1.5' }}>
                {maintenance.recommendations}
              </div>
              {maintenance.next_inspection_cycles > 0 && (
                <div style={{ marginTop: '10px', fontSize: '0.75rem', color: '#888' }}>
                  Next inspection in: <span className="text-cyan">{maintenance.next_inspection_cycles} cycles</span>
                </div>
              )}
            </div>
          )}

          {/* TELEMETRY */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', marginTop: '5px' }}>
            <h3 className="text-cyan" style={{ borderBottom: '1px solid rgba(0,243,255,0.3)', paddingBottom: '10px', margin: 0, fontSize: '0.9rem', letterSpacing: '1.5px' }}>PREDICTION METRICS</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <MetricCard
                label="PREDICTED RUL"
                value={getRULStatus().value}
                unit="cycles"
                status={getRULStatus().status}
                description="Remaining Useful Life"
              />
              <MetricCard
                label="ENGINE STATE"
                value={getStateStatus().value}
                unit=""
                status={getStateStatus().status}
                description="Current Health Status"
              />
              <MetricCard
                label="CONFIDENCE"
                value={prediction?.confidence?.toUpperCase() || '--'}
                unit=""
                status={prediction?.confidence === 'high' ? 'nominal' : prediction?.confidence === 'low' ? 'danger' : 'warning'}
                description="Reliability"
              />
              <MetricCard
                label="UNIT ID"
                value={selectedEngine || '--'}
                unit=""
                status="nominal"
                description="Unit"
              />
            </div>
          </div>
        </div>

        {/* RIGHT PANEL: OUTPUT */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px', minHeight: 0 }}>

          {/* REPORT SCREEN */}
          <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
            <TerminalOutput answer={answer} loading={loading || predicting} prediction={prediction} />
          </div>



          {/* COMMAND CONSOLE */}
          <form onSubmit={handleCommand} className="glow-box" style={{
            display: 'flex',
            gap: '0',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '1px solid var(--neon-cyan)'
          }}>
            <div style={{
              padding: '18px 25px',
              background: 'rgba(0, 243, 255, 0.1)',
              color: 'var(--neon-cyan)',
              fontFamily: 'Orbitron',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <span>COMMAND</span>
              <span style={{ animation: 'blink 1s infinite' }}>_</span>
            </div>
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ask about engine health, predictions, maintenance..."
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                color: '#fff',
                padding: '18px',
                outline: 'none',
                fontFamily: 'monospace',
                fontSize: '1.1rem'
              }}
            />
            <button
              type="submit"
              disabled={loading || !healthStatus?.models_loaded}
              style={{
                background: 'var(--neon-cyan)',
                color: '#000',
                border: 'none',
                padding: '0 40px',
                fontFamily: 'Orbitron',
                fontWeight: 'bold',
                cursor: loading ? 'wait' : 'pointer',
                letterSpacing: '2px',
                transition: 'all 0.3s',
                opacity: !healthStatus?.models_loaded ? 0.5 : 1
              }}
            >
              EXECUTE
            </button>
          </form>
        </div>

      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
    </div>
  )
}
