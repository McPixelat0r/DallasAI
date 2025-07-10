function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
  }
  
  function simulateROI() {
    const roiData = {
      costReduction: Math.floor(Math.random() * 30 + 10),
      revenueIncrease: Math.floor(Math.random() * 50 + 20),
      efficiencyBoost: Math.floor(Math.random() * 25 + 5),
    };
  
    const totalROI = roiData.costReduction + roiData.revenueIncrease + roiData.efficiencyBoost;
  
    const output = `
      📉 Cost Reduction: ${roiData.costReduction}%<br>
      💰 Revenue Increase: ${roiData.revenueIncrease}%<br>
      ⚙️ Efficiency Boost: ${roiData.efficiencyBoost}%<br><br>
      📈 <strong>Total Estimated ROI: ${totalROI}%</strong>
    `;
  
    document.getElementById('roiResult').innerHTML = output;
  }
  