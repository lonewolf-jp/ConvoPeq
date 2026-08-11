// 一時解析スクリプト: trigger_symbol_usage_report.json の blocked 詳細確認用
const fs = require('fs');
const r = JSON.parse(fs.readFileSync('evidence/trigger_symbol_usage_report.json', 'utf8'));
console.log('totalMatches=' + r.totalMatches + ' blockedMatches=' + r.blockedMatches);
console.log('--- symbolStats ---');
r.symbolStats.forEach(s => console.log(s.symbol + ': total=' + s.totalMatches + ' blocked=' + s.blockedMatches));
console.log('--- activeDSP blocked ---');
r.blocked.filter(x => x.symbol === 'activeDSP').forEach(x => console.log(x.path + ':' + x.line + ' allowed=' + x.allowed));
